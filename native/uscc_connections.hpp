// Stream Slater--Condon connections and the first-order commutator. No
// determinant Hamiltonian, candidate columns, or screening cache is retained.
#include <unordered_map>
#include <array>

class ConnectedScreening {
    struct Single { int p,q,target,sign; double base; };
    struct Double { int target; double value; };
    struct Opposite { int p,q; double value; };
    struct Move { int target,sign; };
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
    double h_bound_=0.,a_bound_=0.,maximum_l1_=1.;
    double connection_cutoff_=0.,connection_error_=0.;
    py::array pointers_,rows_,values_;
    bool index32_;
    std::vector<uint64_t> strings_,bits_;
    std::vector<double> phase_,eri_,diagonal_,coulomb_;
    std::vector<std::vector<int>> occupied_;
    std::vector<std::vector<Single>> singles_;
    std::vector<std::vector<Double>> doubles_;
    std::vector<std::vector<Double>> strong_doubles_;
    std::vector<std::vector<Opposite>> strong_opposite_;
    std::vector<Move> single_moves_;
    std::vector<int> addresses_;
    std::unordered_map<uint64_t,int> large_addresses_;
    std::vector<Offset> row_pointers_;
    std::vector<int> columns_;
    std::vector<double> coefficients_;
    std::vector<double> csf_l1_;
    std::vector<Rule> rules_;
    Offset rule_words_=0;
    // Applicability factorizes over alpha/beta strings. Four short bitmaps
    // replace a scan over every excitation word for each occupied sector row.
    std::vector<uint64_t> alpha_forward_,beta_forward_,alpha_reverse_,beta_reverse_;
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
    void hamiltonian(const SparseState &in,SparseState &out,Offset start=0,bool strong=false) const {
        for (Offset position=start;position<static_cast<Offset>(in.touched.size());++position) {
            const auto row=in.touched[position];
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
                const double element=s.sign*(s.base+coulomb_[(s.p*norb_+s.q)*width_+b]);
                if (!strong || std::abs(element)>=connection_cutoff_) out.add(target,v*element*phase_[target]);
            }
            for (const auto &s:singles_[b]) {
                const Offset target=static_cast<Offset>(a)*width_+s.target;
                const double element=s.sign*(s.base+coulomb_[(s.p*norb_+s.q)*width_+a]);
                if (!strong || std::abs(element)>=connection_cutoff_) out.add(target,v*element*phase_[target]);
            }
            const auto &doubles=strong?strong_doubles_:doubles_;
            for (const auto &d:doubles[a]) {
                const Offset target=static_cast<Offset>(d.target)*width_+b;
                out.add(target,v*d.value*phase_[target]);
            }
            for (const auto &d:doubles[b]) {
                const Offset target=static_cast<Offset>(a)*width_+d.target;
                out.add(target,v*d.value*phase_[target]);
            }
            if (strong) {
                for (const auto &s:singles_[a]) {
                    const auto &connections=strong_opposite_[s.p*norb_+s.q];
                    if (connections.size()<singles_[b].size()) {
                        for (const auto &t:connections) {
                            const auto move=single_moves_[(static_cast<Offset>(b)*norb_+t.p)*norb_+t.q];
                            if (move.target<0) continue;
                            const Offset target=static_cast<Offset>(s.target)*width_+move.target;
                            out.add(target,v*s.sign*move.sign*t.value*phase_[target]);
                        }
                    } else {
                        for (const auto &t:singles_[b]) {
                            const double element=eri(s.p,s.q,t.p,t.q);
                            if (std::abs(element)<connection_cutoff_) continue;
                            const Offset target=static_cast<Offset>(s.target)*width_+t.target;
                            out.add(target,v*s.sign*t.sign*element*phase_[target]);
                        }
                    }
                }
            } else {
                for (const auto &s:singles_[a]) for (const auto &t:singles_[b]) {
                    const Offset target=static_cast<Offset>(s.target)*width_+t.target;
                    out.add(target,v*s.sign*t.sign*eri(s.p,s.q,t.p,t.q)*phase_[target]);
                }
            }
        }
    }
    void generator(const SparseState &in,SparseState &out,double scale=1.,Offset start=0) const {
        for (Offset position=start;position<static_cast<Offset>(in.touched.size());++position) {
            const auto row=in.touched[position];
            const double v=scale*in.values[row];
            if (!v) continue;
            const auto bits=bits_[row];
            const int a=row/width_,b=row%width_;
            const Offset alpha_start=static_cast<Offset>(a)*rule_words_,beta_start=static_cast<Offset>(b)*rule_words_;
            for (Offset word=0;word<rule_words_;++word) {
                const uint64_t forward=alpha_forward_[alpha_start+word]&beta_forward_[beta_start+word];
                uint64_t active=forward|(alpha_reverse_[alpha_start+word]&beta_reverse_[beta_start+word]);
                while (active) {
                    const int bit=__builtin_ctzll(active);
                    const auto &r=rules_[word*64+bit];
                    const bool is_forward=(forward&(uint64_t{1}<<bit))!=0;
                    const int sign=1-2*((__builtin_popcountll((is_forward?bits:bits^r.flip)&r.parity)^r.constant)&1);
                    const Offset target=static_cast<Offset>(address(strings_[a]^r.alpha))*width_+address(strings_[b]^r.beta);
                    out.add(target,(is_forward?1.:-1.)*v*r.value*sign);
                    active &= active-1;
                }
            }
        }
    }
    void load_column(Offset column,SparseState &out) const {
        const auto *data=static_cast<const double*>(values_.data());
        for (Offset j=index(pointers_,column);j<index(pointers_,column+1);++j)
            out.add(index(rows_,j),data[j]);
    }
    static std::pair<Offset,double> bounded_tail(SparseState &state,double budget) {
        // Reorder only the sparse support; retain all coefficients for exact
        // refinement. No extra sector buffer or amplitude cutoff is needed.
        if (!(budget>0.)) return {0,0.};
        std::sort(state.touched.begin(),state.touched.end(),[&](Offset a,Offset b) {
            const double av=std::abs(state.values[a]),bv=std::abs(state.values[b]);
            return av<bv || (av==bv && a<b);
        });
        double norm=0.;Offset start=0;
        for (auto row:state.touched) {
            const double next=std::hypot(norm,state.values[row]);
            if (next>budget) break;
            norm=next;++start;
        }
        return {start,norm};
    }
    double exact_linear_element(Offset column,SparseState &input,SparseState &scratch,
                                const SparseState &h,const SparseState &a) const {
        // H+[H,A] is symmetric: <a|HA|i>=<H a|A i> and
        // -<a|AH|i>=<A a|H i>. A few sparse actions on the target CSF
        // resolve an uncertain threshold without rebuilding either product.
        input.clear();load_column(column,input);
        double value=0.;
        for (auto row:input.touched) value+=input.values[row]*h.values[row];
        scratch.clear();hamiltonian(input,scratch);
        for (auto row:a.touched) value+=a.values[row]*scratch.values[row];
        scratch.clear();generator(input,scratch);
        for (auto row:scratch.touched) value+=scratch.values[row]*h.values[row];
        scratch.clear();
        return value;
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
        double scale=0.;
        for (Offset i=0;i<one.size();++i) scale=std::max(scale,std::abs(h[i]));
        for (Offset i=0;i<eri_input.size();++i) scale=std::max(scale,std::abs(e[i]));
        connection_cutoff_=1e-10*scale;
        eri_.assign(e,e+eri_input.size());
        const auto *s=static_cast<const int64_t*>(strings.data()),*bs=static_cast<const int64_t*>(bits.data());
        const auto *ps=static_cast<const double*>(phases.data());
        strings_.assign(s,s+width_);bits_.assign(bs,bs+dimension_);phase_.assign(ps,ps+dimension_);
        if (norb_<=20) addresses_.assign(Offset{1}<<norb_,-1);
        int electrons=__builtin_popcountll(strings_[0]);
        const double singles=electrons*(norb_-electrons);
        const double doubles=electrons*(electrons-1)*(norb_-electrons)*(norb_-electrons-1)/4.;
        // Every row and column has at most this many off-diagonal SD moves.
        // Thus ||H-H_strong||_2 <= sqrt(||D||_1 ||D||_infty) <= degree*cutoff.
        connection_error_=(2*singles+2*doubles+singles*singles)*connection_cutoff_*(1+1e-10);
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
        strong_doubles_.resize(width_);strong_opposite_.resize(norb_*norb_);
        single_moves_.assign(static_cast<Offset>(width_)*norb_*norb_,Move{-1,0});
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
                single_moves_[(static_cast<Offset>(i)*norb_+p)*norb_+q]={address(target),single_sign(strings_[i],q,p)};
            }
            for (size_t q=0;q<occupied_[i].size();++q) for (size_t s=q+1;s<occupied_[i].size();++s)
                for (size_t p=0;p<empty.size();++p) for (size_t r=p+1;r<empty.size();++r) {
                    const int qo=occupied_[i][q],so=occupied_[i][s],po=empty[p],ro=empty[r];
                    const auto mid=strings_[i]^(uint64_t{1}<<qo)^(uint64_t{1}<<po);
                    const int sign=single_sign(strings_[i],qo,po)*single_sign(mid,so,ro);
                    const double value=sign*(eri(po,qo,ro,so)-eri(po,so,ro,qo));
                    if (value) doubles_[i].push_back({address(mid^(uint64_t{1}<<so)^(uint64_t{1}<<ro)),value});
                    if (value && std::abs(value)>=connection_cutoff_)
                        strong_doubles_[i].push_back({address(mid^(uint64_t{1}<<so)^(uint64_t{1}<<ro)),value});
                }
        }
        for (int p=0;p<norb_;++p) for (int q=0;q<norb_;++q) if (p!=q)
            for (int r=0;r<norb_;++r) for (int s=0;s<norb_;++s) if (r!=s) {
                const double value=eri(p,q,r,s);
                if (value && std::abs(value)>=connection_cutoff_) strong_opposite_[p*norb_+q].push_back({r,s,value});
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
        csf_l1_.resize(count_,0.);
        for (Offset i=0;i<count_;++i) {
            double norm=0.;Offset previous=-1;
            for (Offset j=index(pointers_,i);j<index(pointers_,i+1);++j) {
                auto row=index(rows_,j);
                if (row<=previous) throw py::value_error("CSF rows must be sorted and unique");
                previous=row;norm+=data[j]*data[j];
                csf_l1_[i]+=std::abs(data[j]);
                auto at=next[row]++;columns_[at]=i;coefficients_[at]=data[j];
            }
            if (std::abs(norm-1.)>1e-10) throw py::value_error("Expected normalized CSFs");
            maximum_l1_=std::max(maximum_l1_,csf_l1_[i]);
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
        rule_words_=(rules_.size()+63)/64;
        const Offset bitmap_size=static_cast<Offset>(width_)*rule_words_;
        alpha_forward_.assign(bitmap_size,0);beta_forward_.assign(bitmap_size,0);
        alpha_reverse_.assign(bitmap_size,0);beta_reverse_.assign(bitmap_size,0);
        for (Offset k=0;k<static_cast<Offset>(rules_.size());++k) {
            const auto &r=rules_[k];
            uint64_t ah=0,bh=0;
            for (int p=0;p<norb_;++p) {
                ah|=((r.source>>(2*norb_-2*p-1))&1)<<p;
                bh|=((r.source>>(2*norb_-2*p-2))&1)<<p;
            }
            const uint64_t bit=uint64_t{1}<<(k%64);
            for (int i=0;i<width_;++i) {
                const Offset at=static_cast<Offset>(i)*rule_words_+k/64;
                const auto alpha=strings_[i]&r.alpha,beta=strings_[i]&r.beta;
                if (alpha==ah) alpha_forward_[at]|=bit;
                if (beta==bh) beta_forward_[at]|=bit;
                if (alpha==(ah^r.alpha)) alpha_reverse_[at]|=bit;
                if (beta==(bh^r.beta)) beta_reverse_[at]|=bit;
            }
        }
    }
    template<bool Profile> py::array_t<bool> screen_impl(py::array selected,py::array weights,double threshold,bool linear,int threads,double *timings) const {
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
        std::vector<std::array<double,7>> statistics(Profile?workers:0);
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
                double tick=0.;
                auto mark=[&](int phase) {
                    if constexpr(Profile) {
                        const double now=omp_get_wtime();
                        statistics[omp_get_thread_num()][phase]+=now-tick;
                        tick=now;
                    }
                };
                #pragma omp for schedule(dynamic,1)
                for (Offset cursor=0;cursor<selected.size();++cursor) {
                    const Offset i=order[cursor];
                    if (!w[i] || w[i]*bound<threshold) continue;
                    load_column(ids[i],input);
                    if constexpr(Profile) tick=omp_get_wtime();
                    hamiltonian(input,h);
                    mark(0);
                    double operator_error=0.;
                    if (linear) {
                        generator(input,a);
                        mark(1);
                        const double target=threshold/w[i];
                        const auto a_tail=bounded_tail(a,target/(16*h_bound_));
                        const auto h_tail=bounded_tail(h,target/(16*a_bound_));
                        const bool strong=connection_error_*a_bound_<=target/16;
                        operator_error=(h_bound_*a_tail.second+a_bound_*h_tail.second+
                                        (strong?connection_error_*a_bound_:0.))*(1+1e-10);
                        mark(6);
                        hamiltonian(a,r,a_tail.first,strong);
                        mark(2);
                        for (auto row:h.touched) r.add(row,h.values[row]);
                        generator(h,r,-1.,h_tail.first);
                        mark(3);
                    }
                    const auto &value=linear?r:h;
                    // The discarded determinant components contribute at
                    // most cutoff*||C_a||_1 to any CSF. Most commutator tails
                    // are tiny; project large components first and resolve
                    // only uncertain threshold decisions with an exact dot.
                    const double cutoff=linear?threshold/(8*w[i]*maximum_l1_):0.;
                    for (auto row:value.touched) {
                        const double v=value.values[row];
                        if (!v || std::abs(v)<cutoff) continue;
                        for (Offset j=row_pointers_[row];j<row_pointers_[row+1];++j)
                            if (!passed[columns_[j]].load(std::memory_order_relaxed))
                                projected.add(columns_[j],coefficients_[j]*v);
                    }
                    for (auto row:projected.touched) {
                        if (passed[row].load(std::memory_order_relaxed)) continue;
                        const double score=std::abs(projected.values[row])*w[i];
                        const double error=(operator_error+cutoff*csf_l1_[row]*(1+1e-10))*w[i];
                        if (score>=threshold+error) passed[row].store(true,std::memory_order_relaxed);
                        else if (score+error>=threshold) {
                            const double exact=exact_linear_element(row,input,r,h,a);
                            if (std::abs(exact)*w[i]>=threshold) passed[row].store(true,std::memory_order_relaxed);
                        }
                    }
                    mark(4);
                    input.clear();h.clear();a.clear();r.clear();projected.clear();
                    mark(5);
                }
            }
            for (Offset j=0;j<count_;++j) out[j]=passed[j].load(std::memory_order_relaxed);
            for (Offset i=0;i<selected.size();++i) out[ids[i]]=false;
        }
        if constexpr(Profile) for (const auto &entry:statistics) for (int k=0;k<7;++k) timings[k]+=entry[k];
        return result;
    }
    py::array_t<bool> screen(py::array selected,py::array weights,double threshold,bool linear,int threads) const {
        return screen_impl<false>(selected,weights,threshold,linear,threads,nullptr);
    }
    py::tuple profile_screen(py::array selected,py::array weights,double threshold,bool linear,int threads) const {
        py::array_t<double> times(7);
        std::fill(times.mutable_data(),times.mutable_data()+7,0.);
        auto result=screen_impl<true>(selected,weights,threshold,linear,threads,times.mutable_data());
        return py::make_tuple(result,times);
    }
};

static void bind_connections(py::module_ &module) {
    py::class_<ConnectedScreening>(module,"ConnectedScreening")
        .def(py::init<py::array,py::array,py::array,py::array,py::array,py::array,py::array,py::array,py::array,py::array>())
        .def("screen",&ConnectedScreening::screen,py::arg("indices").noconvert(),py::arg("weights").noconvert(),py::arg("threshold"),py::arg("linear")=false,py::arg("threads")=1)
        .def("profile_screen",&ConnectedScreening::profile_screen,py::arg("indices").noconvert(),py::arg("weights").noconvert(),py::arg("threshold"),py::arg("linear")=false,py::arg("threads")=1);
}
