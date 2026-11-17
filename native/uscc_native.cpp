// Optional OpenMP sweeps over an immutable determinant-pair rotation plan.
// No fast math, gate reordering, or Python work in parallel regions.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace py = pybind11;
using Complex = std::complex<double>;
using Offset = py::ssize_t;

#ifdef USCC_PYSCF_BATCH
// PySCF's real direct_spin1 contraction. Link only when both libraries share
// the same OpenMP runtime; the outer team then serializes its nested regions.
extern "C" void FCIcontract_2e_spin1(double*,double*,double*,int,int,int,int,int,int*,int*);
#endif

template<class A,class B> static inline double real_conjugate_product(A a,B b) {
    if constexpr (std::is_same_v<A,double> && std::is_same_v<B,double>) return a*b;
    else if constexpr (std::is_same_v<A,double>) return a*b.real();
    else if constexpr (std::is_same_v<B,double>) return a.real()*b;
    else return a.real()*b.real()+a.imag()*b.imag();
}

static void array_check(const py::array &a, const py::dtype &dtype,
                        Offset size, bool writable=false) {
    if (a.ndim()!=1 || a.size()!=size || !a.dtype().equal(dtype) ||
        !(a.flags() & py::array::c_style) || (writable && !a.writeable()))
        throw py::value_error("Expected a contiguous vector with matching size/dtype and writable output");
    const size_t alignment=std::min<size_t>(a.itemsize(),alignof(double));
    if (reinterpret_cast<uintptr_t>(a.data())%alignment)
        throw py::value_error("Expected aligned array storage");
}

struct Gate { Offset parameter, start, stop; double scale; Offset product=-1; };

#ifdef USCC_PYSCF_BATCH
static py::array_t<double> contract_2e_batch(py::array eri,py::array states,
                                            py::array links,int norb,int threads) {
    if (norb<1 || norb>63 || threads<1 || links.ndim()!=3 || links.shape(2)!=4 ||
        links.shape(0)<1 || links.shape(0)>std::numeric_limits<int>::max() ||
        links.shape(1)>std::numeric_limits<int>::max() || !(links.flags() & py::array::c_style))
        throw py::value_error("Invalid orbital/thread count or triangular link table");
    const int na=static_cast<int>(links.shape(0)),nl=static_cast<int>(links.shape(1));
    const Offset dimension=static_cast<Offset>(na)*na,ntri=norb*(norb+1)/2;
    if (dimension>std::numeric_limits<int>::max() || states.ndim()!=2 || states.shape(1)!=dimension ||
        !(states.flags() & py::array::c_style) || eri.ndim()!=2 || eri.shape(0)!=ntri || eri.shape(1)!=ntri ||
        !(eri.flags() & py::array::c_style))
        throw py::value_error("Expected packed integrals and contiguous row CI vectors");
    array_check(eri.reshape({eri.size()}),py::dtype::of<double>(),ntri*ntri);
    array_check(states.reshape({states.size()}),py::dtype::of<double>(),states.size());
    array_check(links.reshape({links.size()}),py::dtype::of<int>(),links.size());
    auto *l=static_cast<int*>(const_cast<void*>(links.data()));
    for (Offset i=0;i<links.size();i+=4)
        if (l[i]<0 || l[i]>=ntri || l[i+2]<0 || l[i+2]>=na || (l[i+3]!=1 && l[i+3]!=-1))
            throw py::value_error("Invalid triangular link address, orbital index or sign");
    py::array_t<double> output({states.shape(0),dimension});
    auto *e=static_cast<double*>(const_cast<void*>(eri.data()));
    auto *s=static_cast<double*>(const_cast<void*>(states.data()));
    auto *out=output.mutable_data();
    const Offset count=states.shape(0);
    if (!count) return output;
    {
        py::gil_scoped_release release;
        const int levels=omp_get_max_active_levels();
        omp_set_max_active_levels(1);
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (Offset i=0;i<count;++i)
            FCIcontract_2e_spin1(e,s+i*dimension,out+i*dimension,norb,na,na,nl,nl,l,l);
        omp_set_max_active_levels(levels);
    }
    return output;
}
#endif

// Each independent monomial table can be built by a separate worker. NumPy
// buffers are allocated with the GIL held; workers only fill raw pointers.
template<class Index>
static py::list build_pairs_impl(py::array indices,py::array sorted_indices,
                                 py::array index_order,py::array metadata,int threads) {
    if (threads<1) throw py::value_error("Invalid thread count");
    const Offset dimension=indices.size();
    array_check(indices,py::dtype::of<int64_t>(),dimension);
    array_check(sorted_indices,py::dtype::of<int64_t>(),dimension);
    array_check(index_order,py::dtype::of<Offset>(),dimension);
    if (metadata.ndim()!=2 || metadata.shape(1)!=5 ||
        !metadata.dtype().equal(py::dtype::of<uint64_t>()) || !(metadata.flags() & py::array::c_style))
        throw py::value_error("Expected contiguous uint64 monomial metadata with five columns");
    if (reinterpret_cast<uintptr_t>(metadata.data())%alignof(uint64_t))
        throw py::value_error("Expected aligned monomial metadata");
    const Offset count=metadata.shape(0);
    const auto *rules=static_cast<const uint64_t*>(metadata.data());
    const auto *index=static_cast<const int64_t*>(indices.data());
    const auto *sorted=static_cast<const int64_t*>(sorted_indices.data());
    const auto *order=static_cast<const Offset*>(index_order.data());
    for (Offset i=0;i<dimension;++i)
        if (order[i]<0 || order[i]>=dimension || (i && sorted[i]<=sorted[i-1]))
            throw py::value_error("Invalid sorted determinant lookup");
    struct Output {Index *source,*target;int8_t *sign;Offset size;};
    std::vector<Output> outputs;
    py::list result;
    for (Offset k=0;k<count;++k) {
        const uint64_t *rule=rules+5*k;
        if (rule[4]>static_cast<uint64_t>(dimension) || rule[3]>1 || (rule[0]&rule[1])!=rule[0])
            throw py::value_error("Invalid monomial mask or pair count");
        const Offset size=static_cast<Offset>(rule[4]);
        py::array_t<Index> source(size),target(size);
        py::array_t<int8_t> sign(size);
        outputs.push_back({source.mutable_data(),target.mutable_data(),sign.mutable_data(),size});
        result.append(py::make_tuple(source,target,sign));
    }
    std::atomic<bool> invalid{false};
    {
        py::gil_scoped_release release;
        #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
        for (Offset k=0;k<count;++k) {
            const uint64_t *rule=rules+5*k;
            const auto &output=outputs[k];
            Offset cursor=0;
            for (Offset source=0;source<dimension;++source) {
                const uint64_t bits=static_cast<uint64_t>(index[source]);
                if ((bits&rule[1])!=rule[0]) continue;
                const int64_t destination=static_cast<int64_t>(bits^rule[1]);
                const auto *position=std::lower_bound(sorted,sorted+dimension,destination);
                if (cursor>=output.size || position==sorted+dimension || *position!=destination) {
                    invalid.store(true,std::memory_order_relaxed);continue;
                }
                output.source[cursor]=source;
                output.target[cursor]=order[position-sorted];
                output.sign[cursor]=1.-2*((__builtin_popcountll(bits&rule[2])^rule[3])&1);
                ++cursor;
            }
            if (cursor!=output.size) invalid.store(true,std::memory_order_relaxed);
        }
    }
    if (invalid.load()) throw py::value_error("Monomial count mismatch or excitation leaves the determinant sector");
    return result;
}

static py::list build_pairs(py::array indices,py::array sorted_indices,
                            py::array index_order,py::array metadata,int threads) {
    // Match the portable pair cache: addresses need only represent the sector
    // dimension, and fermionic signs are exactly +/-1. Avoid wide temporary
    // tables as well as retaining them alongside the packed rotation plan.
    if (indices.size()<=std::numeric_limits<int32_t>::max())
        return build_pairs_impl<int32_t>(indices,sorted_indices,index_order,metadata,threads);
    return build_pairs_impl<int64_t>(indices,sorted_indices,index_order,metadata,threads);
}

class RotationPlan {
    // Pair buffers must remain immutable for the plan lifetime. Holding array
    // references prevents normal NumPy resizing while the GIL is released.
    py::array sources_, targets_, signs_;
    std::vector<Gate> gates_;
    Offset dimension_, parameters_;
    bool int32_;
    struct Product {
        std::vector<Offset> source[2],target[2];
        std::vector<int8_t> sign[2];
    };
    std::vector<Product> products_;
    Offset width_=0;

    template<class State,class Costate,bool Reverse>
    static inline double rotate_pair(State *state,Costate *costate,Offset source,
                                     Offset target,double sign,double c,double sn) {
        const double ss=sign*sn;
        const State left=state[source],right=state[target];
        if constexpr (Reverse) {
            const Costate cleft=costate[source],cright=costate[target];
            const double derivative=sign*(real_conjugate_product(cright,left)-real_conjugate_product(cleft,right));
            state[source]=c*left+ss*right;
            state[target]=-ss*left+c*right;
            costate[source]=c*cleft+ss*cright;
            costate[target]=-ss*cleft+c*cright;
            return derivative;
        } else {
            state[source]=c*left-ss*right;
            state[target]=ss*left+c*right;
            return 0.;
        }
    }
    template<class Index> void validate_pairs() {
        const auto *source=static_cast<const Index*>(sources_.data());
        const auto *target=static_cast<const Index*>(targets_.data());
        const auto *sign=static_cast<const int8_t*>(signs_.data());
        std::set<std::pair<Offset,Offset>> checked;
        std::vector<Offset> seen(dimension_,-1);
        Offset stamp=0;
        for (const auto &gate:gates_) {
            if (!checked.emplace(gate.start,gate.stop).second) continue;
            for (Offset p=gate.start;p<gate.stop;++p) {
                const Offset s=source[p],t=target[p];
                if (s<0 || t<0 || s>=dimension_ || t>=dimension_ || s==t ||
                    (sign[p]!=1 && sign[p]!=-1))
                    throw py::value_error("Invalid determinant pair address or sign");
                if (seen[s]==stamp || seen[t]==stamp)
                    throw py::value_error("Pairs within a gate must be disjoint");
                seen[s]=seen[t]=stamp;
            }
            ++stamp;
        }
    }

    std::pair<std::vector<double>,std::vector<double>> angles(const py::array &theta,
                                                            bool adjoint) const {
        array_check(theta,py::dtype::of<double>(),parameters_);
        const auto *t=static_cast<const double*>(theta.data());
        for (Offset p=0;p<parameters_;++p)
            if (!std::isfinite(t[p])) throw py::value_error("Parameters must be finite");
        std::vector<double> cosine(gates_.size()),sine(gates_.size());
        for (size_t g=0;g<gates_.size();++g) {
            double angle=t[gates_[g].parameter]*gates_[g].scale*(adjoint?-1.:1.);
            cosine[g]=std::cos(angle);sine[g]=std::sin(angle);
        }
        return {std::move(cosine),std::move(sine)};
    }

    static bool overlap(const py::array &a,const py::array &b) {
        if (!a.nbytes() || !b.nbytes()) return false;
        const auto x=reinterpret_cast<uintptr_t>(a.data());
        const auto y=reinterpret_cast<uintptr_t>(b.data());
        return x<y+b.nbytes() && y<x+a.nbytes();
    }

    template<class State,class Costate,class Index,bool Reverse>
    void sweep(State *state,Costate *costate,double *gradient,
               const std::vector<double> &cosine,const std::vector<double> &sine,
               int threads,Offset minimum,bool adjoint) const {
        const auto *sources=width_?nullptr:static_cast<const Index*>(sources_.data());
        const auto *targets=width_?nullptr:static_cast<const Index*>(targets_.data());
        const auto *signs=width_?nullptr:static_cast<const int8_t*>(signs_.data());
        const Offset count=static_cast<Offset>(gates_.size());
        // Each worker owns a full gradient row: no per-gate reduction/barrier
        // beyond the barrier already required by state dependencies.
        std::vector<double> partials(Reverse?parameters_*threads:0,0.);
        auto apply=[&](Offset g,Offset begin,Offset end,double *local_gradient) {
            const auto &gate=gates_[g];
            const double c=cosine[g],sn=sine[g];
            double derivative=0.;
            if (gate.product>=0) {
                const auto &map=products_[gate.product];
                const Offset nb=map.source[1].size();
                // Divide only at row boundaries. The inner beta loop shares
                // alpha addresses/signs and reads tiny maps instead of the
                // expanded Cartesian determinant-pair tables.
                if (nb) for (Offset a=begin/nb;a*nb<end;++a) {
                    const Offset first=std::max<Offset>(0,begin-a*nb),last=std::min(nb,end-a*nb);
                    const Offset source=map.source[0][a]*width_,target=map.target[0][a]*width_;
                    const double sign=map.sign[0][a];
                    #pragma omp simd reduction(+:derivative)
                    for (Offset b=first;b<last;++b)
                        derivative+=rotate_pair<State,Costate,Reverse>(state,costate,
                            source+map.source[1][b],target+map.target[1][b],sign*map.sign[1][b],c,sn);
                }
            } else {
                // Validation proves every pair is disjoint.
                #pragma omp simd reduction(+:derivative)
                for (Offset p=begin;p<end;++p)
                    derivative+=rotate_pair<State,Costate,Reverse>(state,costate,
                        sources[p],targets[p],signs[p],c,sn);
            }
            if constexpr (Reverse) local_gradient[gate.parameter]+=2*gate.scale*derivative;
        };
        auto gate_index=[&](Offset step) { return Reverse || adjoint?count-1-step:step; };
        bool large=false;
        for (const auto &gate:gates_) large|=gate.stop-gate.start>=minimum;
        if (threads==1 || !large) {
            for (Offset step=0;step<count;++step) {
                const Offset g=gate_index(step);
                apply(g,gates_[g].start,gates_[g].stop,gradient);
            }
            return;
        }
        // One OpenMP region per complete sweep. Small consecutive gates use
        // one serial batch and one barrier, rather than one barrier per gate.
        #pragma omp parallel num_threads(threads)
        {
            const int worker=omp_get_thread_num(),workers=omp_get_num_threads();
            double *local=nullptr;
            if constexpr (Reverse) if (parameters_) local=partials.data()+worker*parameters_;
            for (Offset step=0;step<count;) {
                Offset g=gate_index(step);
                const auto &gate=gates_[g];
                if (gate.stop-gate.start<minimum) {
                    Offset end=step+1;
                    while (end<count) {
                        const auto &next=gates_[gate_index(end)];
                        if (next.stop-next.start>=minimum) break;
                        ++end;
                    }
                    if (worker==0) for (Offset k=step;k<end;++k) {
                        const Offset i=gate_index(k);
                        apply(i,gates_[i].start,gates_[i].stop,local);
                    }
                    step=end;
                } else {
                    const Offset pairs=gate.stop-gate.start;
                    const Offset first=gate.start+pairs*worker/workers;
                    const Offset last=gate.start+pairs*(worker+1)/workers;
                    apply(g,first,last,local);
                    ++step;
                }
                #pragma omp barrier
            }
        }
        if constexpr (Reverse) for (int worker=0;worker<threads;++worker)
            for (Offset p=0;p<parameters_;++p) gradient[p]+=partials[worker*parameters_+p];
    }

    template<class State,class Costate,bool Reverse>
    void dispatch_index(State *state,Costate *costate,double *gradient,
                        const std::vector<double> &cosine,const std::vector<double> &sine,
                        int threads,Offset minimum,bool adjoint) const {
        if (int32_) sweep<State,Costate,int32_t,Reverse>(state,costate,gradient,cosine,sine,threads,minimum,adjoint);
        else sweep<State,Costate,int64_t,Reverse>(state,costate,gradient,cosine,sine,threads,minimum,adjoint);
    }

public:
    RotationPlan(Offset width,Offset parameters,py::array ids,py::array scales,
                 py::array products,py::list maps):dimension_(0),parameters_(parameters),int32_(false),width_(width) {
        if (width<1 || width>std::numeric_limits<Offset>::max()/width || parameters<0)
            throw py::value_error("Invalid string/parameter dimension");
        dimension_=width*width;
        const Offset n=ids.size();
        array_check(ids,py::dtype::of<Offset>(),n);
        array_check(scales,py::dtype::of<double>(),n);
        array_check(products,py::dtype::of<Offset>(),n);
        for (auto item:maps) {
            const auto arrays=py::cast<py::tuple>(item);
            if (arrays.size()!=6) throw py::value_error("Expected six spin-map arrays");
            Product map;
            bool disjoint=false;
            for (int spin=0;spin<2;++spin) {
                const auto source=py::cast<py::array>(arrays[3*spin]);
                const auto target=py::cast<py::array>(arrays[3*spin+1]);
                const auto signs=py::cast<py::array>(arrays[3*spin+2]);
                array_check(source,py::dtype::of<Offset>(),source.size());
                array_check(target,py::dtype::of<Offset>(),source.size());
                array_check(signs,py::dtype::of<int8_t>(),source.size());
                const auto *s=static_cast<const Offset*>(source.data()),*t=static_cast<const Offset*>(target.data());
                const auto *v=static_cast<const int8_t*>(signs.data());
                std::set<Offset> left,right;
                map.source[spin].reserve(source.size());map.target[spin].reserve(source.size());map.sign[spin].reserve(source.size());
                for (Offset i=0;i<source.size();++i) {
                    if (s[i]<0 || s[i]>=width || t[i]<0 || t[i]>=width || (v[i]!=1 && v[i]!=-1) ||
                        !left.insert(s[i]).second || !right.insert(t[i]).second)
                        throw py::value_error("Invalid or duplicate spin-map address/sign");
                    map.source[spin].push_back(s[i]);map.target[spin].push_back(t[i]);map.sign[spin].push_back(v[i]);
                }
                bool overlap=false;
                for (auto address:left) overlap|=right.count(address)!=0;
                disjoint|=!overlap;
            }
            if (!disjoint) throw py::value_error("Cartesian rotation pairs must be disjoint");
            products_.push_back(std::move(map));
        }
        const auto *p=static_cast<const Offset*>(ids.data()),*m=static_cast<const Offset*>(products.data());
        const auto *c=static_cast<const double*>(scales.data());
        for (Offset g=0;g<n;++g) {
            if (p[g]<0 || p[g]>=parameters || m[g]<0 || m[g]>=static_cast<Offset>(products_.size()) || !std::isfinite(c[g]))
                throw py::value_error("Invalid gate parameter, product or scale");
            const auto &map=products_[m[g]];
            gates_.push_back({p[g],0,static_cast<Offset>(map.source[0].size()*map.source[1].size()),c[g],m[g]});
        }
    }

    RotationPlan(Offset dimension,Offset parameters,py::array ids,py::array scales,
                 py::array starts,py::array stops,py::array sources,
                 py::array targets,py::array signs):sources_(sources),targets_(targets),
                 signs_(signs),dimension_(dimension),parameters_(parameters) {
        if (dimension<1 || parameters<0) throw py::value_error("Invalid state/parameter dimension");
        const Offset n=ids.size();
        array_check(ids,py::dtype::of<Offset>(),n);
        array_check(scales,py::dtype::of<double>(),n);
        array_check(starts,py::dtype::of<Offset>(),n);
        array_check(stops,py::dtype::of<Offset>(),n);
        int32_=sources.dtype().equal(py::dtype::of<int32_t>());
        if (!int32_ && !sources.dtype().equal(py::dtype::of<int64_t>()))
            throw py::value_error("Pair addresses must be int32 or int64");
        array_check(sources,sources.dtype(),sources.size());
        array_check(targets,sources.dtype(),sources.size());
        array_check(signs,py::dtype::of<int8_t>(),sources.size());
        if (sources.writeable() || targets.writeable() || signs.writeable())
            throw py::value_error("Pair buffers must be readonly for the plan lifetime");
        const auto *p=static_cast<const Offset*>(ids.data());
        const auto *a=static_cast<const Offset*>(starts.data());
        const auto *b=static_cast<const Offset*>(stops.data());
        const auto *c=static_cast<const double*>(scales.data());
        for (Offset g=0;g<n;++g) {
            if (p[g]<0 || p[g]>=parameters || a[g]<0 || b[g]<a[g] ||
                b[g]>sources.size() || !std::isfinite(c[g]))
                throw py::value_error("Invalid gate parameter, range or scale");
            gates_.push_back({p[g],a[g],b[g],c[g]});
        }
        if (int32_) validate_pairs<int32_t>(); else validate_pairs<int64_t>();
    }

    Offset storage_bytes() const {
        Offset size=gates_.capacity()*sizeof(Gate);
        if (!width_) return size+sources_.nbytes()+targets_.nbytes()+signs_.nbytes();
        for (const auto &map:products_) for (int spin=0;spin<2;++spin)
            size+=(map.source[spin].capacity()+map.target[spin].capacity())*sizeof(Offset)+map.sign[spin].capacity();
        return size;
    }

    void forward(py::array theta,py::array state,int threads,Offset minimum,bool adjoint) const {
        if (threads<1 || minimum<0) throw py::value_error("Invalid thread count or minimum pair count");
        auto [cosine,sine]=angles(theta,adjoint);
        const bool real=state.dtype().equal(py::dtype::of<double>());
        array_check(state,real?py::dtype::of<double>():py::dtype::of<Complex>(),dimension_,true);
        void *data=state.mutable_data();
        py::gil_scoped_release release;
        if (real) dispatch_index<double,double,false>(static_cast<double*>(data),nullptr,nullptr,cosine,sine,threads,minimum,adjoint);
        else dispatch_index<Complex,Complex,false>(static_cast<Complex*>(data),nullptr,nullptr,cosine,sine,threads,minimum,adjoint);
    }

    void forward_batch(py::array theta,py::array states,int threads,bool adjoint) const {
        if (threads<1 || states.ndim()!=2 || states.shape(1)!=dimension_ ||
            !(states.flags() & py::array::c_style))
            throw py::value_error("Expected contiguous row states and a positive thread count");
        const bool real=states.dtype().equal(py::dtype::of<double>());
        array_check(states.reshape({states.size()}),real?py::dtype::of<double>():py::dtype::of<Complex>(),states.size(),true);
        auto [cosine,sine]=angles(theta,adjoint);
        const Offset count=states.shape(0);
        if (!count) return;
        void *data=states.mutable_data();
        py::gil_scoped_release release;
        // Independent states need no synchronization between circuit gates.
        // Keep each complete sweep on its worker, preserving the gate order.
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (Offset i=0;i<count;++i) {
            if (real) dispatch_index<double,double,false>(static_cast<double*>(data)+i*dimension_,nullptr,nullptr,cosine,sine,1,0,adjoint);
            else dispatch_index<Complex,Complex,false>(static_cast<Complex*>(data)+i*dimension_,nullptr,nullptr,cosine,sine,1,0,adjoint);
        }
    }

    void reverse(py::array theta,py::array state,py::array costate,py::array gradient,
                 int threads,Offset minimum) const {
        if (threads<1 || minimum<0) throw py::value_error("Invalid thread count or minimum pair count");
        auto [cosine,sine]=angles(theta,false);
        const bool sr=state.dtype().equal(py::dtype::of<double>()),cr=costate.dtype().equal(py::dtype::of<double>());
        array_check(state,sr?py::dtype::of<double>():py::dtype::of<Complex>(),dimension_,true);
        array_check(costate,cr?py::dtype::of<double>():py::dtype::of<Complex>(),dimension_,true);
        array_check(gradient,py::dtype::of<double>(),parameters_,true);
        if (overlap(state,costate) || overlap(state,gradient) || overlap(costate,gradient))
            throw py::value_error("State, costate and gradient outputs must not overlap");
        void *s=state.mutable_data(),*c=costate.mutable_data();
        auto *g=static_cast<double*>(gradient.mutable_data());
        py::gil_scoped_release release;
        if (sr && cr) dispatch_index<double,double,true>(static_cast<double*>(s),static_cast<double*>(c),g,cosine,sine,threads,minimum,false);
        else if (sr) dispatch_index<double,Complex,true>(static_cast<double*>(s),static_cast<Complex*>(c),g,cosine,sine,threads,minimum,false);
        else if (cr) dispatch_index<Complex,double,true>(static_cast<Complex*>(s),static_cast<double*>(c),g,cosine,sine,threads,minimum,false);
        else dispatch_index<Complex,Complex,true>(static_cast<Complex*>(s),static_cast<Complex*>(c),g,cosine,sine,threads,minimum,false);
    }
};

#include "uscc_scalar.hpp"

PYBIND11_MODULE(_uscc_native,module) {
    bind_scalar(module);
    module.doc()="Persistent-team OpenMP determinant rotations; immutable validated plans";
    module.attr("api_version")=1;
#ifdef USCC_NATIVE_SOURCE_SHA256
    module.attr("source_sha256")=USCC_NATIVE_SOURCE_SHA256;
#endif
#ifdef USCC_NATIVE_OPENMP_RUNTIME
    module.attr("openmp_runtime")=USCC_NATIVE_OPENMP_RUNTIME;
#endif
    module.attr("compiler")=__VERSION__;
    py::class_<RotationPlan>(module,"RotationPlan")
        .def_property_readonly("storage_bytes", &RotationPlan::storage_bytes)
        .def(py::init<Offset,Offset,py::array,py::array,py::array,py::list>(),
             py::arg("width"),py::arg("parameter_count"),py::arg("parameter_ids").noconvert(),
             py::arg("scales").noconvert(),py::arg("product_ids").noconvert(),py::arg("maps"))
        .def(py::init<Offset,Offset,py::array,py::array,py::array,py::array,py::array,py::array,py::array>(),
             py::arg("dimension"),py::arg("parameter_count"),py::arg("parameter_ids").noconvert(),
             py::arg("scales").noconvert(),py::arg("starts").noconvert(),py::arg("stops").noconvert(),
             py::arg("sources").noconvert(),py::arg("targets").noconvert(),py::arg("signs").noconvert())
        .def("forward",&RotationPlan::forward,py::arg("parameters").noconvert(),py::arg("state").noconvert(),
             py::arg("threads")=1,py::arg("minimum_pairs")=0,py::arg("adjoint")=false)
        .def("forward_batch",&RotationPlan::forward_batch,py::arg("parameters").noconvert(),py::arg("states").noconvert(),
             py::arg("threads")=1,py::arg("adjoint")=false)
        .def("reverse",&RotationPlan::reverse,py::arg("parameters").noconvert(),py::arg("state").noconvert(),
             py::arg("costate").noconvert(),py::arg("gradient").noconvert(),py::arg("threads")=1,py::arg("minimum_pairs")=0);
    module.def("openmp_max_threads",[](){return omp_get_max_threads();});
    module.attr("cartesian_rotations")=true;
    module.def("openmp_max_active_levels",[](){return omp_get_max_active_levels();});
#ifdef USCC_PYSCF_BATCH
    module.def("contract_2e_batch",&contract_2e_batch,py::arg("eri").noconvert(),py::arg("states").noconvert(),
               py::arg("links").noconvert(),py::arg("norb"),py::arg("threads")=1);
#endif
    module.def("build_pairs",&build_pairs,py::arg("indices").noconvert(),py::arg("sorted_indices").noconvert(),
               py::arg("index_order").noconvert(),py::arg("metadata").noconvert(),py::arg("threads")=1);
}
