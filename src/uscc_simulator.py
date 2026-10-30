"""Exact simulator acceleration for the existing factorized USCC circuit.

A pure fermionic excitation couples disjoint pairs of determinants. Applying
those signed plane rotations directly avoids its exponentially many Pauli
rotations. Only a fixed (N_alpha,N_beta) sector is stored. This changes neither
component ordering nor the accepted internal Trotter approximation.
"""
from math import comb
import numpy as np
from scipy import sparse
from pyscf import lib
from pyscf.fci import cistring, direct_spin1, spin_op
from excitations import ExcitationTerm


class SectorSimulator:
    def __init__(self, integrals, max_dense_dimension=1024, *, sector_threads=None, use_numba=True, kernels='auto', rotation_threads=None):
        if sector_threads is not None and (type(sector_threads) is not int or sector_threads < 1):
            raise ValueError('sector_threads must be a positive integer or None')
        self.sector_threads = sector_threads
        if kernels not in ('auto','native','numba','numpy'):
            raise ValueError('kernels must be auto, native, numba or numpy')
        if rotation_threads is not None and (type(rotation_threads) is not int or rotation_threads < 1):
            raise ValueError('rotation_threads must be a positive integer or None')
        self.rotation_threads = rotation_threads
        self.integrals = integrals
        self.n_modes = integrals.qubits
        self.n_orbitals = self.n_modes//2
        self.n_alpha = self.n_beta = integrals.electrons//2
        self.dimension = comb(self.n_orbitals, self.n_alpha)**2
        self._native_module = None
        self._forward = self._reverse = None
        # Preserve use_numba=False as the NumPy oracle for existing callers.
        if kernels == 'auto' and not use_numba:
            kernels = 'numpy'
        if kernels in ('auto','native'):
            try:
                import _uscc_native
                if getattr(_uscc_native,'api_version',None) != 1:
                    raise ImportError('Native module API mismatch; rebuild the extension')
                self._native_module = _uscc_native
            except ImportError as error:
                if kernels == 'native':
                    raise RuntimeError('Native kernels unavailable; build with python native/build.py') from error
        if self._native_module is None and kernels in ('auto','numba'):
            from uscc_kernels import forward, reverse
            self._forward, self._reverse = forward, reverse
        self.kernel_backend = 'native' if self._native_module is not None else 'numba' if self._forward is not None else 'numpy'
        strings = cistring.make_strings(range(self.n_orbitals), self.n_alpha)
        # PySCF uses alpha-major, beta-minor CI coefficients with all alpha
        # creators before beta creators. Our JW convention interleaves spins.
        interleaved = np.zeros(len(strings),dtype=np.int64)
        parity = np.zeros((len(strings),len(strings)),dtype=np.int8)
        lower_beta_parity = np.zeros(len(strings),dtype=np.int8)
        for p in range(self.n_orbitals):
            occupied = ((strings >> p) & 1).astype(np.int8)
            interleaved |= occupied.astype(np.int64) << (self.n_modes-2*p-2)
            parity ^= occupied[:,None]*lower_beta_parity[None,:]
            lower_beta_parity ^= occupied
        self.indices = ((interleaved[:,None] << 1) | interleaved[None,:]).ravel()
        self._index_order = np.argsort(self.indices)
        self._sorted_indices = self.indices[self._index_order]
        self.address = {int(index): i for i,index in enumerate(self.indices)}
        self.phases = (1.-2*parity).ravel()
        # Stored g[p,q,r,s] = (p s|q r), matching OpenFermion a†p a†q ar as.
        self.chemist_eri = np.ascontiguousarray(integrals.two_body.transpose(0,3,1,2))
        self.shape = (len(strings),len(strings))
        self.hamiltonian = None
        self.effective_eri = None
        if self.dimension <= max_dense_dimension:
            with lib.with_omp_threads(self.sector_threads):
                addresses, matrix = direct_spin1.pspace(integrals.one_body, self.chemist_eri, self.n_orbitals, (self.n_alpha,self.n_beta), np=self.dimension)
            if not np.array_equal(addresses,np.arange(self.dimension)):
                raise RuntimeError('Unexpected PySCF determinant ordering')
            self.hamiltonian = self.phases[:,None]*matrix*self.phases[None,:]
            self.hamiltonian.flat[::self.dimension+1] += integrals.constant
        else:
            with lib.with_omp_threads(self.sector_threads):
                self.effective_eri = direct_spin1.absorb_h1e(integrals.one_body, self.chemist_eri, self.n_orbitals, (self.n_alpha,self.n_beta), .5)
            links = cistring.gen_linkstr_index_trilidx(range(self.n_orbitals),self.n_alpha)
            self._link_index = (links,links)
        self.initial = np.zeros(self.dimension)
        hf_index = ((1 << integrals.electrons)-1) << (self.n_modes-integrals.electrons)
        self.initial[self.address[hf_index]] = 1.
        self._pairs = {}
        self._plan = ()
        self._packed_plan = None
        self._adjoint_plan = None
        self._native_plan = None
        self._excitations = None
        self._steps = None
        self._observables = {}

    def apply_hamiltonian(self, state):
        if self.hamiltonian is not None:
            return self.hamiltonian @ state
        ci = (self.phases*state).reshape(self.shape)
        # PySCF's real-integral contraction also handles complex vectors via
        # separate real/imaginary contractions (never a dense sector matrix).
        def contract(part):
            return np.asarray(direct_spin1.contract_2e(self.effective_eri, np.ascontiguousarray(part), self.n_orbitals, (self.n_alpha,self.n_beta), link_index=self._link_index)).ravel()
        # None honors PySCF's ambient OpenMP setting. Explicit counts are scoped
        # to these calls, including both components of a complex state.
        with lib.with_omp_threads(self.sector_threads):
            applied = contract(ci.real)
            if np.iscomplexobj(ci):
                applied = applied + 1j*contract(ci.imag)
        return self.phases*applied + self.integrals.constant*state

    def pairs(self, holes, particles):
        key = holes, particles
        if key not in self._pairs:
            term = ExcitationTerm(holes,particles)
            if max(holes+particles) >= self.n_modes:
                raise ValueError('Fermion mode outside the register')
            hole_mask = sum(1 << (self.n_modes-h-1) for h in holes)
            particle_mask = sum(1 << (self.n_modes-p-1) for p in particles)
            flip_mask = hole_mask | particle_mask
            source = np.flatnonzero((self.indices & flip_mask) == hole_mask)
            destinations = self.indices[source] ^ flip_mask
            positions = np.searchsorted(self._sorted_indices,destinations)
            if np.any(positions == self.dimension) or np.any(self._sorted_indices[positions] != destinations):
                raise ValueError('Excitation leaves the fixed spin/electron sector')
            target = self._index_order[positions]
            # Fermion signs are parity of a fixed mask of the initial index,
            # plus the signs caused by earlier toggles in the operator word.
            parity_mask, toggled, constant = 0, 0, 0
            for mode in holes+tuple(reversed(particles)):
                prefix = ((1 << self.n_modes)-1) ^ ((1 << (self.n_modes-mode))-1)
                parity_mask ^= prefix
                constant ^= (toggled & prefix).bit_count() & 1
                toggled ^= 1 << (self.n_modes-mode-1)
            parity = self.indices[source] & parity_mask
            # XOR folding supports NumPy versions without bitwise_count.
            for shift in (32,16,8,4,2,1):
                parity ^= parity >> shift
            signs = 1.-2*((parity ^ constant) & 1)
            self._pairs[key] = (source,target,signs)
        return self._pairs[key]

    def apply_hamiltonian_batch(self, vectors):
        """Independent row contractions, with a bounded outer native team.

        Real direct_spin1 calls use one worker each rather than synchronizing
        a team inside every contraction. Small batches and unsupported builds
        retain the scalar oracle; complex inputs split into real components.
        """
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension or not np.all(np.isfinite(vectors)):
            raise ValueError('Expected finite row vectors matching the electron sector')
        vectors = np.asarray(vectors, dtype=complex if np.iscomplexobj(vectors) else float)
        if not len(vectors):
            return np.empty(vectors.shape, dtype=vectors.dtype)
        if self.hamiltonian is not None:
            return vectors @ self.hamiltonian.T
        threads = self.sector_threads or lib.num_threads()
        batch = None if self._native_module is None else getattr(self._native_module, 'contract_2e_batch', None)
        if batch is None or threads < 8 or len(vectors) < max(8, threads//2) or vectors.dtype not in (np.dtype('float64'), np.dtype('complex128')):
            return np.array([self.apply_hamiltonian(v) for v in vectors]).reshape(vectors.shape)
        ci = vectors*self.phases
        links = self._link_index[0]
        if np.iscomplexobj(ci):
            real = batch(self.effective_eri, np.ascontiguousarray(ci.real), links, self.n_orbitals, min(threads,len(ci)))
            imag = batch(self.effective_eri, np.ascontiguousarray(ci.imag), links, self.n_orbitals, min(threads,len(ci)))
            applied = real+1j*imag
        else:
            applied = batch(self.effective_eri, np.ascontiguousarray(ci), links, self.n_orbitals, min(threads,len(ci)))
        return applied*self.phases+self.integrals.constant*vectors

    def plan(self, excitations, steps):
        if self._steps != steps or self._excitations is None or len(excitations) != len(self._excitations) or any(a is not b for a,b in zip(excitations,self._excitations)):
            if not isinstance(steps,int) or steps < 1:
                raise ValueError('trotter_steps must be a positive integer')
            if self._native_module is not None:
                self._build_native_pairs(excitations)
            self._plan = tuple((i,term.coefficient/steps,*self.pairs(term.holes,term.particles)) for i,excitation in enumerate(excitations) for _ in range(steps) for term in excitation.terms)
            self._packed_plan = None
            self._adjoint_plan = None
            self._native_plan = None
            # Keep references so object IDs cannot be recycled under the cache.
            self._excitations, self._steps = tuple(excitations), steps
        return self._plan

    def _build_native_pairs(self, excitations):
        missing = dict.fromkeys((term.holes,term.particles) for excitation in excitations
                                for term in excitation.terms if (term.holes,term.particles) not in self._pairs)
        if not missing:
            return
        metadata = []
        for holes,particles in missing:
            if max(holes+particles) >= self.n_modes:
                raise ValueError('Fermion mode outside the register')
            if sum(h%2 for h in holes) != sum(p%2 for p in particles):
                raise ValueError('Excitation leaves the fixed spin/electron sector')
            hole_mask = sum(1 << (self.n_modes-h-1) for h in holes)
            flip_mask = hole_mask | sum(1 << (self.n_modes-p-1) for p in particles)
            parity_mask,toggled,constant = 0,0,0
            for mode in holes+tuple(reversed(particles)):
                prefix = ((1 << self.n_modes)-1) ^ ((1 << (self.n_modes-mode))-1)
                parity_mask ^= prefix
                constant ^= (toggled & prefix).bit_count() & 1
                toggled ^= 1 << (self.n_modes-mode-1)
            counts = []
            for spin,occupied in ((0,self.n_alpha),(1,self.n_beta)):
                nh = sum(h%2==spin for h in holes)
                npart = sum(p%2==spin for p in particles)
                remaining = occupied-nh
                available = self.n_orbitals-nh-npart
                counts.append(comb(available,remaining) if 0 <= remaining <= available else 0)
            metadata.append((hole_mask,flip_mask,parity_mask,constant,counts[0]*counts[1]))
        threads = self.sector_threads if self.sector_threads is not None else lib.num_threads()
        arrays = self._native_module.build_pairs(self.indices,self._sorted_indices,self._index_order,
                                                 np.array(metadata,dtype=np.uint64),threads)
        self._pairs.update(zip(missing,arrays))

    def packed_plan(self):
        if self._packed_plan is None:
            parameter_ids, scales, starts, stops = [], [], [], []
            sources, targets, signs, offsets = [], [], [], {}
            offset = 0
            for parameter,scale,source,target,sign in self._plan:
                # Reuse pair storage across spin couplings and Trotter steps.
                key = id(source)
                if key not in offsets:
                    offsets[key] = offset, offset+len(source)
                    offset += len(source)
                    sources.append(source); targets.append(target); signs.append(sign)
                start,stop = offsets[key]
                parameter_ids.append(parameter); scales.append(scale)
                starts.append(start); stops.append(stop)
            def concatenate(arrays,dtype):
                return np.concatenate(arrays).astype(dtype,copy=False) if arrays else np.empty(0,dtype=dtype)
            # Reduce pair-table storage for large sectors. Addresses fit in
            # int32 for ordinary sector sizes and signs are exactly ±1;
            # keep offsets in native integers for plans exceeding 2**31 pairs.
            address_dtype = np.int32 if self.dimension <= np.iinfo(np.int32).max else np.int64
            self._packed_plan = (np.array(parameter_ids,dtype=int),np.array(scales),np.array(starts,dtype=int),np.array(stops,dtype=int),concatenate(sources,address_dtype),concatenate(targets,address_dtype),concatenate(signs,np.int8))
        return self._packed_plan

    def native_plan(self):
        """Validated immutable native plan, rebuilt after order/growth/step changes."""
        if self._native_module is None:
            raise RuntimeError('Native kernels are not selected')
        if self._native_plan is None:
            packed = self.packed_plan()
            for array in packed:
                array.flags.writeable = False
            self._native_plan = self._native_module.RotationPlan(self.dimension,len(self._excitations),*packed)
        return self._native_plan

    def effective_rotation_threads(self):
        if self.rotation_threads is not None:
            return self.rotation_threads
        # Parallel barriers cost more than the pair work in small sectors.
        # The measured large-sector default is conservative; explicit counts
        # permit tuning on machines/workloads with different scaling.
        available = self.sector_threads if self.sector_threads is not None else lib.num_threads()
        return min(4,available) if self.dimension >= 200000 else 1

    @staticmethod
    def rotate(state, source, target, cosine, signed_sine):
        left, right = state[source].copy(), state[target].copy()
        state[source] = cosine*left-signed_sine*right
        state[target] = signed_sine*left+cosine*right

    def apply_unitary(self, vector, parameters, excitations, steps=1, *, adjoint=False):
        """Linear U or U† action on an arbitrary, unnormalized sector vector."""
        parameters = np.asarray(parameters,dtype=float)
        if parameters.shape != (len(excitations),) or not np.all(np.isfinite(parameters)):
            raise ValueError('Expected one finite parameter per excitation')
        vector = np.asarray(vector)
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError('Expected a finite vector matching the determinant sector')
        state = np.array(vector,dtype=np.result_type(vector,float),copy=True)
        plan = self.plan(excitations,steps)
        if self._native_module is not None and state.dtype in (np.dtype('float64'),np.dtype('complex128')):
            self.native_plan().forward(np.ascontiguousarray(parameters),state,self.effective_rotation_threads(),adjoint=adjoint)
            return state
        if self._forward is not None and state.dtype in (np.dtype('float64'),np.dtype('complex128')):
            packed = self.packed_plan()
            if adjoint:
                if self._adjoint_plan is None:
                    ids,scales,starts,stops,sources,targets,signs = packed
                    self._adjoint_plan = (ids[::-1].copy(),-scales[::-1].copy(),starts[::-1].copy(),stops[::-1].copy(),sources,targets,signs)
                packed = self._adjoint_plan
            self._forward(parameters,state,*packed)
            return state
        if adjoint:
            plan = reversed(plan)
        for i,scale,source,target,sign in plan:
            angle = (-1 if adjoint else 1)*parameters[i]*scale
            self.rotate(state,source,target,np.cos(angle),sign*np.sin(angle))
        return state

    def apply_unitary_batch(self, vectors, parameters, excitations, steps=1, *, adjoint=False):
        """Apply the same circuit to independent row vectors, without normalization.

        Native workers own whole states, eliminating per-gate synchronization.
        Callers bound batch memory; older extensions and Python kernels fall
        back to the same scalar operation. Hamiltonian threads set batch workers.
        """
        vectors = np.asarray(vectors)
        parameters = np.asarray(parameters, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension or not np.all(np.isfinite(vectors)):
            raise ValueError('Expected finite row vectors matching the electron sector')
        if parameters.shape != (len(excitations),) or not np.all(np.isfinite(parameters)):
            raise ValueError('Expected finite parameters matching the excitation list')
        self.plan(excitations, steps)
        states = np.array(vectors, dtype=np.result_type(vectors, float), order='C', copy=True)
        if len(states) == 1:
            states[0] = self.apply_unitary(states[0], parameters, excitations, steps, adjoint=adjoint)
            return states
        if self._native_module is not None and hasattr(self.native_plan(), 'forward_batch') and states.dtype in (np.dtype('float64'),np.dtype('complex128')):
            threads = min(len(states), self.sector_threads or lib.num_threads())
            if threads:
                self.native_plan().forward_batch(np.ascontiguousarray(parameters), states, threads, adjoint=adjoint)
        else:
            for i in range(len(states)):
                states[i] = self.apply_unitary(states[i], parameters, excitations, steps, adjoint=adjoint)
        return states

    def state(self, parameters, excitations, steps=1, initial=None):
        return self.apply_unitary(self.initial if initial is None else initial,parameters,excitations,steps)

    def energy_and_gradient(self, parameters, excitations, steps=1, hamiltonian=None):
        state = self.state(parameters,excitations,steps)
        costate = self.apply_hamiltonian(state) if hamiltonian is None else self.observable_matrix(hamiltonian) @ state
        # Long factorized circuits accumulate small norm errors. The Rayleigh
        # quotient and its centered derivative avoid magnifying those errors
        # by the absolute electronic energy near an optimizer minimum.
        norm = float(np.vdot(state,state).real)
        energy = float(np.vdot(state,costate).real)/norm
        costate = (costate-energy*state)/norm
        gradient = np.zeros(len(parameters))
        if self._native_module is not None and state.dtype in (np.dtype('float64'),np.dtype('complex128')) and costate.dtype in (np.dtype('float64'),np.dtype('complex128')):
            self.native_plan().reverse(np.ascontiguousarray(parameters,dtype=float),state,costate,gradient,self.effective_rotation_threads())
            return energy,gradient
        if self._reverse is not None and state.dtype in (np.dtype('float64'),np.dtype('complex128')) and costate.dtype in (np.dtype('float64'),np.dtype('complex128')):
            self._reverse(np.asarray(parameters,dtype=float),state,costate,gradient,*self.packed_plan())
            return energy,gradient
        # Reverse both the state and costate: no tape or per-gate state storage.
        # dU/dtheta = scale*(tau-tau†)*U, evaluated before undoing each gate.
        for i,scale,source,target,sign in reversed(self.plan(excitations,steps)):
            derivative = np.vdot(costate[target],sign*state[source])-np.vdot(costate[source],sign*state[target])
            gradient[i] += 2*scale*derivative.real
            angle = -parameters[i]*scale
            cosine, sine = np.cos(angle), sign*np.sin(angle)
            self.rotate(state,source,target,cosine,sine)
            self.rotate(costate,source,target,cosine,sine)
        return energy,gradient

    def project_input(self, vector):
        """Return sector coefficients, or None to request full-register fallback."""
        if not sparse.issparse(vector) or vector.shape not in ((2**self.n_modes,1),(1,2**self.n_modes)):
            raise ValueError('Expected a sparse normalized row or column matching the register')
        if vector.format not in ('csr','csc') or len(vector.indptr) != 2:
            raise ValueError('Use a CSC column or CSR row')
        vector = vector.copy()
        vector.sum_duplicates()
        vector.eliminate_zeros()
        if not np.isclose(np.linalg.norm(vector.data),1.):
            raise ValueError('Prepared state must be normalized')
        if any(int(index) not in self.address for index in vector.indices):
            return None
        initial = np.zeros(self.dimension,dtype=vector.dtype)
        for index,value in zip(vector.indices,vector.data):
            initial[self.address[int(index)]] = value
        return initial

    def full_state(self, sector_state):
        state = np.zeros(2**self.n_modes,dtype=sector_state.dtype)
        state[self.indices] = sector_state
        return state

    def observable_matrix(self, operator):
        # Public arbitrary-observable support, cached independently of the
        # integral Hamiltonian. Off-sector output is irrelevant to <psi|O|psi>.
        key = operator.hash
        cached = self._observables.get(key)
        if cached is None:
            import pennylane as qml
            try:
                full = sparse.csr_array(operator.sparse_matrix(wire_order=range(self.n_modes)))
            except qml.operation.SparseMatrixUndefinedError:
                full = sparse.csr_array(qml.matrix(operator,wire_order=range(self.n_modes)))
            matrix = full[np.ix_(self.indices,self.indices)]
            self._observables[key] = (operator,matrix)
            return matrix
        return cached[1]

    def expectation(self, state, operator):
        return np.real_if_close(np.vdot(state,self.observable_matrix(operator) @ state)).item()

    def spin_squared(self, state):
        # Apply spin ladders in adjacent spin sectors, avoiding construction of
        # an S² matrix and Python fermion-word loops over every determinant.
        ci = np.ascontiguousarray((self.phases*state).reshape(self.shape))
        with lib.with_omp_threads(self.sector_threads):
            applied = spin_op.contract_ss(ci.real,self.n_orbitals,(self.n_alpha,self.n_beta))
            if np.iscomplexobj(ci):
                applied = applied+1j*spin_op.contract_ss(ci.imag,self.n_orbitals,(self.n_alpha,self.n_beta))
        return float(np.vdot(ci,applied).real)
