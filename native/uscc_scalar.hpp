// Exact scalar-block sweeps. Included after the shared pybind validation helpers.
#include <chrono>

struct ScalarEdge { Offset row,column; double value; };
struct ScalarSpectrum {
    py::array generator, frequencies, cosine, sine;
    Offset size, modes;
    std::vector<ScalarEdge> edges;
    std::vector<double> evolution(double angle) const {
        std::vector<double> result(size*size,0.);
        for (Offset j=0;j<size;++j) result[j*size+j]=1.;
        const auto *f=static_cast<const double*>(frequencies.data());
        const auto *p=static_cast<const double*>(cosine.data());
        const auto *k=static_cast<const double*>(sine.data());
        for (Offset mode=0;mode<modes;++mode) {
            const double half=std::sin(angle*f[mode]/2.);
            const double c=-2*half*half,s=std::sin(angle*f[mode]);
            for (Offset j=0;j<size*size;++j)
                result[j]+=c*p[mode*size*size+j]+s*k[mode*size*size+j];
        }
        return result;
    }
};

struct ScalarEmbedding {
    py::array alpha,beta,alpha_signs,beta_signs;
    Offset spectrum,size,na,nb,components;
};

struct ScalarFactor {
    std::vector<ScalarSpectrum> spectra;
    std::vector<ScalarEmbedding> blocks;
    Offset copies=0;
    Offset work=0;
};

class ScalarPlan {
    using Matrices=std::vector<std::vector<double>>;
    using Evolution=std::vector<Matrices>;
    Offset dimension_,width_,parameters_,maximum_=0;
    bool int32_;
    std::vector<ScalarFactor> factors_;

    static void readonly(const py::array &a) {
        if (a.writeable()) throw py::value_error("Scalar plan buffers must be readonly");
    }

    template<class Index> void validate_addresses() const {
        std::vector<Offset> seen(dimension_,-1);
        for (Offset parameter=0;parameter<parameters_;++parameter)
            for (const auto &b:factors_[parameter].blocks) {
                const auto *alpha=static_cast<const Index*>(b.alpha.data());
                const auto *beta=static_cast<const Index*>(b.beta.data());
                const auto *as=static_cast<const int8_t*>(b.alpha_signs.data());
                const auto *bs=static_cast<const int8_t*>(b.beta_signs.data());
                for (Offset i=0;i<b.components*b.na*b.size;++i)
                    if (alpha[i]<0 || alpha[i]>=width_ || (as[i]!=1 && as[i]!=-1))
                        throw py::value_error("Invalid scalar alpha address or sign");
                for (Offset i=0;i<b.components*b.nb*b.size;++i)
                    if (beta[i]<0 || beta[i]>=width_ || (bs[i]!=1 && bs[i]!=-1))
                        throw py::value_error("Invalid scalar beta address or sign");
                for (Offset component=0;component<b.components;++component)
                for (Offset a=0;a<b.na;++a) for (Offset beta_row=0;beta_row<b.nb;++beta_row)
                    for (Offset j=0;j<b.size;++j) {
                        const Offset index=alpha[(component*b.na+a)*b.size+j]*width_+beta[(component*b.nb+beta_row)*b.size+j];
                        if (seen[index]==parameter)
                            throw py::value_error("Scalar blocks and spectator copies must be disjoint");
                        seen[index]=parameter;
                    }
            }
    }

    static bool overlap(const py::array &a,const py::array &b) {
        if (!a.nbytes() || !b.nbytes()) return false;
        const auto x=reinterpret_cast<uintptr_t>(a.data()),y=reinterpret_cast<uintptr_t>(b.data());
        return x<y+b.nbytes() && y<x+a.nbytes();
    }

    std::vector<double> parameters(const py::array &theta) const {
        array_check(theta,py::dtype::of<double>(),parameters_);
        const auto *values=static_cast<const double*>(theta.data());
        std::vector<double> result(parameters_);
        for (Offset i=0;i<parameters_;++i) {
            if (!std::isfinite(values[i])) throw py::value_error("Parameters must be finite");
            result[i]=values[i];
        }
        return result;
    }

    Evolution evolution(const std::vector<double> &theta,bool adjoint) const {
        Evolution matrices(parameters_);
        for (Offset i=0;i<parameters_;++i) {
            if (theta[i]==0.) continue;
            for (const auto &s:factors_[i].spectra)
                matrices[i].push_back(s.evolution(adjoint?-theta[i]:theta[i]));
        }
        return matrices;
    }

    template<class State,class Costate,class Index,bool Reverse>
    void sweep(State *state,Costate *costate,double *gradient,
               const std::vector<double> &theta,int threads,bool adjoint,double *timings,
               const Evolution *prepared=nullptr) const {
        const auto matrix_start=std::chrono::steady_clock::now();
        Evolution local;
        if (!prepared) local=evolution(theta,Reverse || adjoint);
        const auto &matrices=prepared?*prepared:local;
        const auto sweep_start=std::chrono::steady_clock::now();
        if (timings) timings[0]=std::chrono::duration<double>(sweep_start-matrix_start).count();
        std::vector<double> partials(Reverse?parameters_*threads:0,0.);
        auto worker=[&](int rank,int workers) {
            std::vector<State> input(maximum_),output(maximum_);
            std::vector<Costate> dual(Reverse?maximum_:0),dual_output(Reverse?maximum_:0);
            std::vector<Offset> addresses(maximum_);
            std::vector<int8_t> signs(maximum_);
            for (Offset step=0;step<parameters_;++step) {
                const Offset i=(Reverse || adjoint)?parameters_-1-step:step;
                if (!Reverse && theta[i]==0.) continue;
                const auto &factor=factors_[i];
                // A 72-state block costs much more than a two-state rotation.
                // Partition estimated arithmetic, not just spectator count.
                const Offset first=factor.work*rank/workers,last=factor.work*(rank+1)/workers;
                Offset offset=0;
                double derivative=0.;
                for (const auto &b:factor.blocks) {
                    const Offset weight=b.size*(b.size+4),copies=b.components*b.na*b.nb;
                    const Offset stop=offset+copies*weight;
                    const Offset begin=std::max(first,offset),end=std::min(last,stop);
                    if (begin<end) {
                        const auto *alpha=static_cast<const Index*>(b.alpha.data());
                        const auto *beta=static_cast<const Index*>(b.beta.data());
                        const auto *as=static_cast<const int8_t*>(b.alpha_signs.data());
                        const auto *bs=static_cast<const int8_t*>(b.beta_signs.data());
                        const auto *g=static_cast<const double*>(factor.spectra[b.spectrum].generator.data());
                        const double *u=theta[i]==0.?nullptr:matrices[i][b.spectrum].data();
                        const Offset copy_begin=(begin-offset+weight-1)/weight;
                        const Offset copy_end=(end-offset+weight-1)/weight;
                        for (Offset copy=copy_begin;copy<copy_end;++copy) {
                            const Offset component=copy/(b.na*b.nb),a=(copy/b.nb)%b.na,beta_row=copy%b.nb,d=b.size;
                            for (Offset j=0;j<d;++j) {
                                addresses[j]=alpha[(component*b.na+a)*d+j]*width_+beta[(component*b.nb+beta_row)*d+j];
                                signs[j]=as[(component*b.na+a)*d+j]*bs[(component*b.nb+beta_row)*d+j];
                                input[j]=static_cast<double>(signs[j])*state[addresses[j]];
                                if constexpr (Reverse) dual[j]=static_cast<double>(signs[j])*costate[addresses[j]];
                            }
                            if constexpr (Reverse) {
                                for (const auto &edge:factor.spectra[b.spectrum].edges)
                                    derivative+=2*edge.value*(real_conjugate_product(dual[edge.row],input[edge.column])
                                                             -real_conjugate_product(dual[edge.column],input[edge.row]));
                            }
                            if (u) {
                                if (d==2) {
                                    output[0]=u[0]*input[0]+u[1]*input[1];
                                    output[1]=u[2]*input[0]+u[3]*input[1];
                                    if constexpr (Reverse) {
                                        dual_output[0]=u[0]*dual[0]+u[1]*dual[1];
                                        dual_output[1]=u[2]*dual[0]+u[3]*dual[1];
                                    }
                                } else for (Offset j=0;j<d;++j) {
                                    State value=0.;
                                    Costate dual_value=0.;
                                    for (Offset k=0;k<d;++k) {
                                        value+=u[j*d+k]*input[k];
                                        if constexpr (Reverse) dual_value+=u[j*d+k]*dual[k];
                                    }
                                    output[j]=value;
                                    if constexpr (Reverse) dual_output[j]=dual_value;
                                }
                            }
                            if (u) for (Offset j=0;j<d;++j) {
                                state[addresses[j]]=static_cast<double>(signs[j])*output[j];
                                if constexpr (Reverse) costate[addresses[j]]=static_cast<double>(signs[j])*dual_output[j];
                            }
                        }
                    }
                    offset=stop;
                }
                if constexpr (Reverse) partials[rank*parameters_+i]+=derivative;
                if (workers>1) {
                    #pragma omp barrier
                }
            }
        };
        if (threads==1) worker(0,1);
        else {
            #pragma omp parallel num_threads(threads)
            worker(omp_get_thread_num(),omp_get_num_threads());
        }
        if constexpr (Reverse) for (int rank=0;rank<threads;++rank)
            for (Offset i=0;i<parameters_;++i) gradient[i]+=partials[rank*parameters_+i];
        if (timings) timings[1]=std::chrono::duration<double>(std::chrono::steady_clock::now()-sweep_start).count();
    }

    template<class State,class Costate,bool Reverse>
    void dispatch(State *state,Costate *costate,double *gradient,
                  const std::vector<double> &theta,int threads,bool adjoint,double *timings=nullptr,
                  const Evolution *prepared=nullptr) const {
        if (int32_) sweep<State,Costate,int32_t,Reverse>(state,costate,gradient,theta,threads,adjoint,timings,prepared);
        else sweep<State,Costate,int64_t,Reverse>(state,costate,gradient,theta,threads,adjoint,timings,prepared);
    }

public:
    ScalarPlan(Offset dimension,Offset width,py::list factors):
        dimension_(dimension),width_(width),parameters_(factors.size()),int32_(dimension<=std::numeric_limits<int32_t>::max()) {
        if (dimension<1 || width<1 || width>dimension || dimension/width!=width || dimension%width)
            throw py::value_error("Invalid scalar sector dimension or string width");
        for (const auto &item:factors) {
            const auto factor=py::cast<py::tuple>(item);
            if (factor.size()!=2) throw py::value_error("Expected scalar spectra and embeddings");
            ScalarFactor f;
            for (const auto &s:py::cast<py::list>(factor[0])) {
                const auto arrays=py::cast<py::tuple>(s);
                if (arrays.size()!=4) throw py::value_error("Expected four spectral buffers");
                ScalarSpectrum spectrum{py::cast<py::array>(arrays[0]),py::cast<py::array>(arrays[1]),
                                        py::cast<py::array>(arrays[2]),py::cast<py::array>(arrays[3]),0,0};
                const auto &g=spectrum.generator;
                if (g.ndim()!=2 || g.shape(0)!=g.shape(1) || g.shape(0)<2)
                    throw py::value_error("Expected a square scalar generator");
                spectrum.size=g.shape(0);spectrum.modes=spectrum.frequencies.size();
                if (spectrum.cosine.ndim()!=3 || spectrum.sine.ndim()!=3 ||
                    spectrum.cosine.shape(0)!=spectrum.modes || spectrum.sine.shape(0)!=spectrum.modes ||
                    spectrum.cosine.shape(1)!=spectrum.size || spectrum.cosine.shape(2)!=spectrum.size ||
                    spectrum.sine.shape(1)!=spectrum.size || spectrum.sine.shape(2)!=spectrum.size)
                    throw py::value_error("Invalid scalar spectral projector shape");
                for (auto a:{g,spectrum.frequencies,spectrum.cosine,spectrum.sine}) {
                    if (!(a.flags() & py::array::c_style)) throw py::value_error("Expected contiguous spectral buffers");
                    array_check(a.reshape({a.size()}),py::dtype::of<double>(),a.size());readonly(a);
                    const auto *v=static_cast<const double*>(a.data());
                    for (Offset j=0;j<a.size();++j)
                        if (!std::isfinite(v[j])) throw py::value_error("Spectral buffers must be finite");
                }
                const auto *values=static_cast<const double*>(g.data());
                for (Offset row=0;row<spectrum.size;++row) for (Offset col=row+1;col<spectrum.size;++col)
                    if (values[row*spectrum.size+col]!=0.)
                        spectrum.edges.push_back({row,col,values[row*spectrum.size+col]});
                f.spectra.push_back(std::move(spectrum));
            }
            for (const auto &b:py::cast<py::list>(factor[1])) {
                const auto arrays=py::cast<py::tuple>(b);
                if (arrays.size()!=5) throw py::value_error("Expected a spectrum index and four embedding buffers");
                ScalarEmbedding block{py::cast<py::array>(arrays[1]),py::cast<py::array>(arrays[2]),
                                      py::cast<py::array>(arrays[3]),py::cast<py::array>(arrays[4]),
                                      py::cast<Offset>(arrays[0]),0,0,0,0};
                if (block.spectrum<0 || block.spectrum>=static_cast<Offset>(f.spectra.size()))
                    throw py::value_error("Invalid scalar spectrum index");
                block.size=f.spectra[block.spectrum].size;
                for (const auto &a:{block.alpha,block.beta,block.alpha_signs,block.beta_signs}) {
                    if (!(a.flags() & py::array::c_style)) throw py::value_error("Expected contiguous embedding buffers");
                    if ((a.ndim()!=2 && a.ndim()!=3) || a.shape(a.ndim()-1)!=block.size || a.shape(0)<1 || a.shape(a.ndim()-2)<1)
                        throw py::value_error("Invalid scalar embedding shape");
                    readonly(a);
                }
                block.components=block.alpha.ndim()==3?block.alpha.shape(0):1;
                block.na=block.alpha.shape(block.alpha.ndim()-2);block.nb=block.beta.shape(block.beta.ndim()-2);
                const Offset beta_components=block.beta.ndim()==3?block.beta.shape(0):1;
                if (beta_components!=block.components || block.alpha_signs.ndim()!=block.alpha.ndim() ||
                    block.beta_signs.ndim()!=block.beta.ndim() || block.alpha_signs.shape(0)!=block.alpha.shape(0) ||
                    block.beta_signs.shape(0)!=block.beta.shape(0) ||
                    block.alpha_signs.size()!=block.alpha.size() || block.beta_signs.size()!=block.beta.size())
                    throw py::value_error("Scalar addresses and signs must match");
                for (auto a:{block.alpha,block.beta})
                    array_check(a.reshape({a.size()}),int32_?py::dtype::of<int32_t>():py::dtype::of<int64_t>(),a.size());
                for (auto a:{block.alpha_signs,block.beta_signs})
                    array_check(a.reshape({a.size()}),py::dtype::of<int8_t>(),a.size());
                maximum_=std::max(maximum_,block.size);
                f.copies+=block.components*block.na*block.nb;
                f.work+=block.components*block.na*block.nb*block.size*(block.size+4);
                f.blocks.push_back(std::move(block));
            }
            factors_.push_back(std::move(f));
        }
        if (int32_) validate_addresses<int32_t>();else validate_addresses<int64_t>();
    }

    void forward(py::array theta,py::array state,int threads,bool adjoint) const {
        if (threads<1) throw py::value_error("Invalid scalar thread count");
        auto values=parameters(theta);
        const bool real=state.dtype().equal(py::dtype::of<double>());
        array_check(state,real?py::dtype::of<double>():py::dtype::of<Complex>(),dimension_,true);
        void *data=state.mutable_data();
        py::gil_scoped_release release;
        if (real) dispatch<double,double,false>(static_cast<double*>(data),nullptr,nullptr,values,threads,adjoint);
        else dispatch<Complex,Complex,false>(static_cast<Complex*>(data),nullptr,nullptr,values,threads,adjoint);
    }

    void forward_batch(py::array theta,py::array states,int threads,bool adjoint) const {
        if (threads<1 || states.ndim()!=2 || states.shape(1)!=dimension_ ||
            !(states.flags() & py::array::c_style))
            throw py::value_error("Expected contiguous scalar row states and a positive thread count");
        const bool real=states.dtype().equal(py::dtype::of<double>());
        array_check(states.reshape({states.size()}),real?py::dtype::of<double>():py::dtype::of<Complex>(),states.size(),true);
        auto values=parameters(theta);
        const Offset count=states.shape(0);
        if (!count) return;
        void *data=states.mutable_data();
        py::gil_scoped_release release;
        const auto matrices=evolution(values,adjoint);
        // A worker owns a complete state, preserving all scalar-factor order.
        // Single-worker sweeps have no barriers or nested OpenMP teams.
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (Offset i=0;i<count;++i) {
            if (real) dispatch<double,double,false>(static_cast<double*>(data)+i*dimension_,nullptr,nullptr,values,1,adjoint,nullptr,&matrices);
            else dispatch<Complex,Complex,false>(static_cast<Complex*>(data)+i*dimension_,nullptr,nullptr,values,1,adjoint,nullptr,&matrices);
        }
    }

    void reverse(py::array theta,py::array state,py::array costate,py::array gradient,int threads,py::object timing_output) const {
        if (threads<1) throw py::value_error("Invalid scalar thread count");
        auto values=parameters(theta);
        const bool sr=state.dtype().equal(py::dtype::of<double>()),cr=costate.dtype().equal(py::dtype::of<double>());
        array_check(state,sr?py::dtype::of<double>():py::dtype::of<Complex>(),dimension_,true);
        array_check(costate,cr?py::dtype::of<double>():py::dtype::of<Complex>(),dimension_,true);
        array_check(gradient,py::dtype::of<double>(),parameters_,true);
        if (overlap(state,costate) || overlap(state,gradient) || overlap(costate,gradient))
            throw py::value_error("Scalar state, costate and gradient outputs must not overlap");
        py::array timing;
        double *timings=nullptr;
        if (!timing_output.is_none()) {
            timing=py::cast<py::array>(timing_output);
            array_check(timing,py::dtype::of<double>(),2,true);
            if (overlap(timing,state) || overlap(timing,costate) || overlap(timing,gradient))
                throw py::value_error("Scalar timing and state/gradient buffers must not overlap");
            timings=static_cast<double*>(timing.mutable_data());
        }
        auto *s=state.mutable_data(),*c=costate.mutable_data();
        auto *g=static_cast<double*>(gradient.mutable_data());
        py::gil_scoped_release release;
        if (sr && cr) dispatch<double,double,true>(static_cast<double*>(s),static_cast<double*>(c),g,values,threads,false,timings);
        else if (sr) dispatch<double,Complex,true>(static_cast<double*>(s),static_cast<Complex*>(c),g,values,threads,false,timings);
        else if (cr) dispatch<Complex,double,true>(static_cast<Complex*>(s),static_cast<double*>(c),g,values,threads,false,timings);
        else dispatch<Complex,Complex,true>(static_cast<Complex*>(s),static_cast<Complex*>(c),g,values,threads,false,timings);
    }
};

static void bind_scalar(py::module_ &module) {
    module.attr("scalar_api_version")=1;
    py::class_<ScalarPlan>(module,"ScalarPlan")
        .def(py::init<Offset,Offset,py::list>(),py::arg("dimension"),py::arg("string_width"),py::arg("factors"))
        .def("forward",&ScalarPlan::forward,py::arg("parameters").noconvert(),py::arg("state").noconvert(),
             py::arg("threads")=1,py::arg("adjoint")=false)
        .def("forward_batch",&ScalarPlan::forward_batch,py::arg("parameters").noconvert(),py::arg("states").noconvert(),
             py::arg("threads")=1,py::arg("adjoint")=false)
        .def("reverse",&ScalarPlan::reverse,py::arg("parameters").noconvert(),py::arg("state").noconvert(),
             py::arg("costate").noconvert(),py::arg("gradient").noconvert(),py::arg("threads")=1,py::arg("timings")=py::none());
}
