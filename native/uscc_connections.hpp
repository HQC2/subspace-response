// Stream Slater--Condon connections and the first-order commutator. No
// determinant Hamiltonian, candidate columns, or screening cache is retained.
#include <unordered_map>

class ConnectedScreening {
    struct Single { int p,q,target,sign; double base; };
    struct Double { int target; double value; };
    struct Rule { uint64_t source,flip,parity,alpha,beta; int constant; double value; };
    struct SparseState {
        std::vector<double> values;
        std::vector<Offset> touched;
        std::vector<unsigned char> present;
        explicit SparseState(Offset n):values(n,0.),present(n,0) {}
        void add(Offset i,double v) {
            if (!v) return;
            if (!present[i]) { present[i]=1; touched.push_back(i); }
            values[i]+=v;
        }
        void clear() {
            for (auto i:touched) {values[i]=0.;present[i]=0;}
            touched.clear();
        }
    };
    int norb_,width_;
    Offset dimension_,count_;
    double h_bound_=0.,a_bound_=0.;
    py::array pointers_,rows_,values_;
    bool index32_;
    std::vector<uint64_t> strings_,bits_;
    std::vector<double> phase_,eri_,diagonal_,coulomb_;
    std::vector<std::vector<int>> occupied_;
    std::vector<std::vector<Single>> singles_;
    std::vector<std::vector<Double>> doubles_;
    std::vector<int> addresses_;
    std::unordered_map<uint64_t,int> large_addresses_;
    std::vector<Offset> row_pointers_;
    std::vector<int> columns_;
    std::vector<double> coefficients_;
    std::vector<Rule> rules_;
    Offset index(const py::array &a,Offset i) const {
        return index32_?static_cast<const int32_t*>(a.data())[i]:static_cast<const int64_t*>(a.data())[i];
    }
    int address(uint64_t s) const {
        if (!addresses_.empty()) return addresses_[s];
        auto found=large_addresses_.find(s);
        return found==large_addresses_.end()?-1:found->second;
    }
    double eri(int p,int q,int r,int s) const {
        return eri_[((p*norb_+q)*norb_+r)*norb_+s];
    }
    static int single_sign(uint64_t s,int q,int p) {
        return 1-2*((__builtin_popcountll(s&((uint64_t{1}<<q)-1))+
                     __builtin_popcountll((s^(uint64_t{1}<<q))&((uint64_t{1}<<p)-1)))&1);
    }
    void hamiltonian(const SparseState &in,SparseState &out) const {
        for (auto row:in.touched) {
            const double v=in.values[row]*phase_[row];
            if (!v) continue;
            const int a=row/width_,b=row%width_;
            // A scalar energy shift has no external CSF couplings and
            // commutes with A. Omit it before either commutator product.
            double d=diagonal_[a]+diagonal_[b];
            for (int p:occupied_[a]) for (int q:occupied_[b]) d+=eri(p,p,q,q);
            out.add(row,v*d*phase_[row]);
            for (const auto &s:singles_[a]) {
                const Offset target=static_cast<Offset>(s.target)*width_+b;
                out.add(target,v*s.sign*(s.base+coulomb_[(s.p*norb_+s.q)*width_+b])*phase_[target]);
            }
            for (const auto &s:singles_[b]) {
                const Offset target=static_cast<Offset>(a)*width_+s.target;
                out.add(target,v*s.sign*(s.base+coulomb_[(s.p*norb_+s.q)*width_+a])*phase_[target]);
            }
            for (const auto &d:doubles_[a]) {
                const Offset target=static_cast<Offset>(d.target)*width_+b;
                out.add(target,v*d.value*phase_[target]);
            }
            for (const auto &d:doubles_[b]) {
                const Offset target=static_cast<Offset>(a)*width_+d.target;
                out.add(target,v*d.value*phase_[target]);
            }
            for (const auto &s:singles_[a]) for (const auto &t:singles_[b]) {
                const Offset target=static_cast<Offset>(s.target)*width_+t.target;
                out.add(target,v*s.sign*t.sign*eri(s.p,s.q,t.p,t.q)*phase_[target]);
            }
        }
    }
    void generator(const SparseState &in,SparseState &out,double scale=1.) const {
        for (auto row:in.touched) {
            const double v=scale*in.values[row];
            if (!v) continue;
            const auto bits=bits_[row];
            const int a=row/width_,b=row%width_;
            for (const auto &r:rules_) {
                const uint64_t masked=bits&r.flip;
                const bool forward=masked==r.source;
                if (!forward && masked!=(r.source^r.flip)) continue;
                const int sign=1-2*((__builtin_popcountll((forward?bits:bits^r.flip)&r.parity)^r.constant)&1);
                const Offset target=static_cast<Offset>(address(strings_[a]^r.alpha))*width_+address(strings_[b]^r.beta);
                out.add(target,(forward?1.:-1.)*v*r.value*sign);
            }
        }
    }
    void load_column(Offset column,SparseState &out) const {
        const auto *data=static_cast<const double*>(values_.data());
        for (Offset j=index(pointers_,column);j<index(pointers_,column+1);++j)
            out.add(index(rows_,j),data[j]);
    }
public:
    ConnectedScreening(py::array one,py::array eri_input,
                      py::array strings,py::array bits,py::array phases,
                      py::array pointers,py::array rows,py::array values,
                      py::array rules,py::array amplitudes):
        pointers_(pointers),rows_(rows),values_(values) {
        if (one.ndim()!=2 || one.shape(0)!=one.shape(1) || one.shape(0)<1 || one.shape(0)>31 || !(one.flags() & py::array::c_style))
            throw py::value_error("Expected one-electron integrals with 1..31 spatial orbitals");
        norb_=one.shape(0); width_=strings.size(); dimension_=static_cast<Offset>(width_)*width_;
        count_=pointers.size()-1;
        if (width_<1 || count_<0 || count_>std::numeric_limits<int>::max() || eri_input.ndim()!=4 ||
            eri_input.shape(0)!=norb_ || eri_input.shape(1)!=norb_ || eri_input.shape(2)!=norb_ || eri_input.shape(3)!=norb_ ||
            !(eri_input.flags() & py::array::c_style))
            throw py::value_error("Invalid connection dimensions");
        array_check(one.reshape({one.size()}),py::dtype::of<double>(),norb_*norb_);
        array_check(eri_input.reshape({eri_input.size()}),py::dtype::of<double>(),static_cast<Offset>(norb_)*norb_*norb_*norb_);
        array_check(strings,py::dtype::of<int64_t>(),width_);
        array_check(bits,py::dtype::of<int64_t>(),dimension_);
        array_check(phases,py::dtype::of<double>(),dimension_);
        array_check(values,py::dtype::of<double>(),rows.size());
        index32_=pointers.dtype().equal(py::dtype::of<int32_t>());
        const auto itype=index32_?py::dtype::of<int32_t>():py::dtype::of<int64_t>();
        array_check(pointers,itype,count_+1);array_check(rows,itype,values.size());
        if (rules.ndim()!=2 || rules.shape(1)!=4 || !(rules.flags() & py::array::c_style)) throw py::value_error("Expected four contiguous generator masks per rule");
        array_check(rules.reshape({rules.size()}),py::dtype::of<uint64_t>(),rules.size());
        array_check(amplitudes,py::dtype::of<double>(),rules.shape(0));
        const auto *h=static_cast<const double*>(one.data());
        const auto *e=static_cast<const double*>(eri_input.data());
        for (Offset i=0;i<one.size();++i) if (!std::isfinite(h[i])) throw py::value_error("Integrals must be finite");
        for (Offset i=0;i<eri_input.size();++i) if (!std::isfinite(e[i])) throw py::value_error("Integrals must be finite");
        for (Offset i=0;i<one.size();++i) h_bound_+=2*std::abs(h[i]);
        for (Offset i=0;i<eri_input.size();++i) h_bound_+=2*std::abs(e[i]);
        eri_.assign(e,e+eri_input.size());
        const auto *s=static_cast<const int64_t*>(strings.data()),*bs=static_cast<const int64_t*>(bits.data());
        const auto *ps=static_cast<const double*>(phases.data());
        strings_.assign(s,s+width_);bits_.assign(bs,bs+dimension_);phase_.assign(ps,ps+dimension_);
        if (norb_<=20) addresses_.assign(Offset{1}<<norb_,-1);
        int electrons=__builtin_popcountll(strings_[0]);
        for (int i=0;i<width_;++i) {
            if (strings_[i]>=(uint64_t{1}<<norb_) || __builtin_popcountll(strings_[i])!=electrons ||
                (i && strings_[i]<=strings_[i-1])) throw py::value_error("Invalid spin strings");
            if (addresses_.empty()) large_addresses_[strings_[i]]=i;else addresses_[strings_[i]]=i;
        }
        for (Offset i=0;i<dimension_;++i) {
            uint64_t expected=0;
            for (int p=0;p<norb_;++p) {
                expected|=((strings_[i/width_]>>p)&1)<<(2*norb_-2*p-1);
                expected|=((strings_[i%width_]>>p)&1)<<(2*norb_-2*p-2);
            }
            if (bits_[i]!=expected || (phase_[i]!=1. && phase_[i]!=-1.)) throw py::value_error("Invalid interleaved determinant map or phases");
        }
        occupied_.resize(width_);singles_.resize(width_);doubles_.resize(width_);diagonal_.resize(width_,0.);
        coulomb_.resize(static_cast<Offset>(norb_)*norb_*width_,0.);
        for (int i=0;i<width_;++i) {
            std::vector<int> empty;
            for (int p=0;p<norb_;++p) (strings_[i]&(uint64_t{1}<<p)?occupied_[i]:empty).push_back(p);
            for (int p:occupied_[i]) {
                diagonal_[i]+=h[p*norb_+p];
                for (int q:occupied_[i]) diagonal_[i]+=.5*(eri(p,p,q,q)-eri(p,q,q,p));
            }
            for (int p=0;p<norb_;++p) for (int q=0;q<norb_;++q)
                for (int k:occupied_[i]) coulomb_[(p*norb_+q)*width_+i]+=eri(p,q,k,k);
            for (int q:occupied_[i]) for (int p:empty) {
                double base=h[p*norb_+q];
                for (int k:occupied_[i]) base+=eri(p,q,k,k)-eri(p,k,k,q);
                auto target=strings_[i]^(uint64_t{1}<<p)^(uint64_t{1}<<q);
                if (address(target)<0) throw py::value_error("Incomplete spin-string basis");
                singles_[i].push_back({p,q,address(target),single_sign(strings_[i],q,p),base});
            }
            for (size_t q=0;q<occupied_[i].size();++q) for (size_t s=q+1;s<occupied_[i].size();++s)
                for (size_t p=0;p<empty.size();++p) for (size_t r=p+1;r<empty.size();++r) {
                    const int qo=occupied_[i][q],so=occupied_[i][s],po=empty[p],ro=empty[r];
                    const auto mid=strings_[i]^(uint64_t{1}<<qo)^(uint64_t{1}<<po);
                    const int sign=single_sign(strings_[i],qo,po)*single_sign(mid,so,ro);
                    const double value=sign*(eri(po,qo,ro,so)-eri(po,so,ro,qo));
                    if (value) doubles_[i].push_back({address(mid^(uint64_t{1}<<so)^(uint64_t{1}<<ro)),value});
                }
        }
        if (index(pointers_,0)!=0 || index(pointers_,count_)!=values.size()) throw py::value_error("Invalid CSF pointers");
        row_pointers_.assign(dimension_+1,0);
        const auto *data=static_cast<const double*>(values.data());
        for (Offset i=0;i<count_;++i) if (index(pointers_,i)>index(pointers_,i+1)) throw py::value_error("Invalid CSF pointers");
        for (Offset i=0;i<rows.size();++i) {
            auto row=index(rows_,i);
            if (row<0 || row>=dimension_ || !std::isfinite(data[i])) throw py::value_error("Invalid CSF row or coefficient");
            ++row_pointers_[row+1];
        }
        for (Offset i=0;i<dimension_;++i) row_pointers_[i+1]+=row_pointers_[i];
        auto next=row_pointers_;
        columns_.resize(rows.size());coefficients_.resize(rows.size());
        for (Offset i=0;i<count_;++i) {
            double norm=0.;Offset previous=-1;
            for (Offset j=index(pointers_,i);j<index(pointers_,i+1);++j) {
                auto row=index(rows_,j);
                if (row<=previous) throw py::value_error("CSF rows must be sorted and unique");
                previous=row;norm+=data[j]*data[j];
                auto at=next[row]++;columns_[at]=i;coefficients_[at]=data[j];
            }
            if (std::abs(norm-1.)>1e-10) throw py::value_error("Expected normalized CSFs");
        }
        const auto *m=static_cast<const uint64_t*>(rules.data());
        const auto *t=static_cast<const double*>(amplitudes.data());
        for (Offset i=0;i<rules.shape(0);++i) {
            Rule rule{m[4*i],m[4*i+1],m[4*i+2],0,0,static_cast<int>(m[4*i+3]),t[i]};
            if ((rule.source&rule.flip)!=rule.source || !rule.flip || rule.flip>=(uint64_t{1}<<(2*norb_)) ||
                rule.parity>=(uint64_t{1}<<(2*norb_)) || m[4*i+3]>1 || !std::isfinite(rule.value))
                throw py::value_error("Invalid generator rule");
            uint64_t alpha_source=0,beta_source=0;
            for (int p=0;p<norb_;++p) {
                rule.alpha|=((rule.flip>>(2*norb_-2*p-1))&1)<<p;
                rule.beta|=((rule.flip>>(2*norb_-2*p-2))&1)<<p;
                alpha_source|=((rule.source>>(2*norb_-2*p-1))&1)<<p;
                beta_source|=((rule.source>>(2*norb_-2*p-2))&1)<<p;
            }
            if (__builtin_popcountll(alpha_source)*2!=__builtin_popcountll(rule.alpha) ||
                __builtin_popcountll(beta_source)*2!=__builtin_popcountll(rule.beta))
                throw py::value_error("Generator leaves the spin sector");
            if (rule.value) {rules_.push_back(rule);a_bound_+=2*std::abs(rule.value);}
        }
    }
    py::array_t<bool> screen(py::array selected,py::array weights,double threshold,bool linear,int threads) const {
        if (threads<1 || !std::isfinite(threshold) || threshold<0) throw py::value_error("Invalid thread count or threshold");
        array_check(selected,py::dtype::of<int64_t>(),selected.size());
        array_check(weights,py::dtype::of<double>(),selected.size());
        const auto *ids=static_cast<const int64_t*>(selected.data());
        const auto *w=static_cast<const double*>(weights.data());
        for (Offset i=0;i<selected.size();++i)
            if (ids[i]<0 || ids[i]>=count_ || w[i]<0 || !std::isfinite(w[i])) throw py::value_error("Invalid response indices or weights");
        py::array_t<bool> result(count_);auto *out=result.mutable_data();std::fill(out,out+count_,threshold==0.);
        for (Offset i=0;i<selected.size();++i) out[ids[i]]=false;
        if (threshold==0.) return result;
        if (!selected.size()) return result;
        linear=linear && !rules_.empty();
        // Triangle/operator-norm bounds can rule out a whole response entry
        // before visiting any connections. This is not amplitude truncation.
        const double bound=h_bound_*(linear?1+2*a_bound_:1.)*(1+1e-10);
        // Dense scratch with sparse touched lists avoids hashing every
        // contribution. Cap the total worker scratch near 128 MiB, except
        // that one worker is always allowed for a larger sector.
        const Offset bytes=(linear?4:2)*dimension_*17+count_*18;
        const int workers=std::min<Offset>({threads,selected.size(),std::max<Offset>(1,128*1024*1024/ std::max<Offset>(1,bytes))});
        std::vector<std::atomic<bool>> passed(count_);
        for (Offset j=0;j<count_;++j) passed[j].store(false,std::memory_order_relaxed);
        for (Offset i=0;i<selected.size();++i) passed[ids[i]].store(true,std::memory_order_relaxed);
        std::vector<Offset> order(selected.size());
        for (Offset i=0;i<selected.size();++i) order[i]=i;
        std::stable_sort(order.begin(),order.end(),[&](Offset a,Offset b){return w[a]>w[b];});
        {
            py::gil_scoped_release release;
            #pragma omp parallel num_threads(workers)
            {
                SparseState input(dimension_),h(dimension_),a(linear?dimension_:0),r(linear?dimension_:0),projected(count_);
                #pragma omp for schedule(dynamic,1)
                for (Offset cursor=0;cursor<selected.size();++cursor) {
                    const Offset i=order[cursor];
                    if (!w[i] || w[i]*bound<threshold) continue;
                    load_column(ids[i],input);
                    hamiltonian(input,h);
                    if (linear) {
                        generator(input,a);
                        hamiltonian(a,r);
                        for (auto row:h.touched) r.add(row,h.values[row]);
                        generator(h,r,-1.);
                    }
                    const auto &value=linear?r:h;
                    for (auto row:value.touched) {
                        const double v=value.values[row];
                        if (!v) continue;
                        for (Offset j=row_pointers_[row];j<row_pointers_[row+1];++j)
                            if (!passed[columns_[j]].load(std::memory_order_relaxed))
                                projected.add(columns_[j],coefficients_[j]*v);
                    }
                    for (auto row:projected.touched)
                        if (!passed[row].load(std::memory_order_relaxed) && std::abs(projected.values[row])*w[i]>=threshold)
                            passed[row].store(true,std::memory_order_relaxed);
                    input.clear();h.clear();a.clear();r.clear();projected.clear();
                }
            }
            for (Offset j=0;j<count_;++j) out[j]=passed[j].load(std::memory_order_relaxed);
            for (Offset i=0;i<selected.size();++i) out[ids[i]]=false;
        }
        return result;
    }
};

static void bind_connections(py::module_ &module) {
    py::class_<ConnectedScreening>(module,"ConnectedScreening")
        .def(py::init<py::array,py::array,py::array,py::array,py::array,py::array,py::array,py::array,py::array,py::array>())
        .def("screen",&ConnectedScreening::screen,py::arg("indices").noconvert(),py::arg("weights").noconvert(),py::arg("threshold"),py::arg("linear")=false,py::arg("threads")=1);
}
