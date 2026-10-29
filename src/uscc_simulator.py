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
from pyscf.fci import cistring, direct_spin1
from excitations import ExcitationTerm, apply_fermion_word


class SectorSimulator:
    def __init__(self, integrals, max_dense_dimension=1024, *, sector_threads=None):
        if sector_threads is not None and (type(sector_threads) is not int or sector_threads < 1):
            raise ValueError('sector_threads must be a positive integer or None')
        self.sector_threads = sector_threads
        self.integrals = integrals
        self.n_modes = integrals.qubits
        self.n_orbitals = self.n_modes//2
        self.n_alpha = self.n_beta = integrals.electrons//2
        self.dimension = comb(self.n_orbitals, self.n_alpha)**2
        strings = cistring.make_strings(range(self.n_orbitals), self.n_alpha)
        determinants, phases = [], []
        # PySCF uses alpha-major, beta-minor CI coefficients with all alpha
        # creators before beta creators. Our JW convention interleaves spins.
        for alpha in strings:
            alpha = int(alpha)
            for beta in strings:
                beta = int(beta)
                index, inversions = 0, 0
                for p in range(self.n_orbitals):
                    if alpha >> p & 1:
                        index |= 1 << (self.n_modes-2*p-1)
                        inversions += (beta & ((1 << p)-1)).bit_count()
                    if beta >> p & 1:
                        index |= 1 << (self.n_modes-2*p-2)
                determinants.append(index)
                phases.append((-1)**inversions)
        self.indices = np.array(determinants, dtype=np.int64)
        self.address = {int(index): i for i,index in enumerate(self.indices)}
        self.phases = np.array(phases, dtype=float)
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
        self.initial = np.zeros(self.dimension)
        hf_index = ((1 << integrals.electrons)-1) << (self.n_modes-integrals.electrons)
        self.initial[self.address[hf_index]] = 1.
        self._pairs = {}
        self._plan = ()
        self._excitations = None
        self._steps = None
        self._spin_matrix = None
        self._observables = {}

    def apply_hamiltonian(self, state):
        if self.hamiltonian is not None:
            return self.hamiltonian @ state
        ci = (self.phases*state).reshape(self.shape)
        # PySCF's real-integral contraction also handles complex vectors via
        # separate real/imaginary contractions (never a dense sector matrix).
        def contract(part):
            return np.asarray(direct_spin1.contract_2e(self.effective_eri, np.ascontiguousarray(part), self.n_orbitals, (self.n_alpha,self.n_beta))).ravel()
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
            source, target, signs = [], [], []
            for col,index in enumerate(self.indices):
                result = term.apply(int(index), self.n_modes)
                if result is not None:
                    destination = self.address.get(result[0])
                    if destination is None:
                        raise ValueError('Excitation leaves the fixed spin/electron sector')
                    source.append(col); target.append(destination); signs.append(result[1])
            self._pairs[key] = (np.array(source,dtype=int),np.array(target,dtype=int),np.array(signs))
        return self._pairs[key]

    def plan(self, excitations, steps):
        if self._steps != steps or self._excitations is None or len(excitations) != len(self._excitations) or any(a is not b for a,b in zip(excitations,self._excitations)):
            if not isinstance(steps,int) or steps < 1:
                raise ValueError('trotter_steps must be a positive integer')
            self._plan = tuple((i,term.coefficient/steps,*self.pairs(term.holes,term.particles)) for i,excitation in enumerate(excitations) for _ in range(steps) for term in excitation.terms)
            # Keep references so object IDs cannot be recycled under the cache.
            self._excitations, self._steps = tuple(excitations), steps
        return self._plan

    @staticmethod
    def rotate(state, source, target, cosine, signed_sine):
        left, right = state[source].copy(), state[target].copy()
        state[source] = cosine*left-signed_sine*right
        state[target] = signed_sine*left+cosine*right

    def state(self, parameters, excitations, steps=1, initial=None):
        parameters = np.asarray(parameters,dtype=float)
        if parameters.shape != (len(excitations),) or not np.all(np.isfinite(parameters)):
            raise ValueError('Expected one finite parameter per excitation')
        state = self.initial.copy() if initial is None else np.array(initial,dtype=np.result_type(initial,float),copy=True)
        for i,scale,source,target,sign in self.plan(excitations,steps):
            angle = parameters[i]*scale
            self.rotate(state,source,target,np.cos(angle),sign*np.sin(angle))
        return state

    def energy_and_gradient(self, parameters, excitations, steps=1, hamiltonian=None):
        state = self.state(parameters,excitations,steps)
        costate = self.apply_hamiltonian(state) if hamiltonian is None else self.observable_matrix(hamiltonian) @ state
        energy = float(np.vdot(state,costate).real)
        gradient = np.zeros(len(parameters))
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
        if self._spin_matrix is None:
            rows,cols,data = [],[],[]
            for col,index in enumerate(self.indices):
                for p in range(self.n_orbitals):
                    plus = apply_fermion_word(int(index),self.n_modes,((2*p,True),(2*p+1,False)))
                    if plus is None:
                        continue
                    for q in range(self.n_orbitals):
                        minus = apply_fermion_word(plus[0],self.n_modes,((2*q+1,True),(2*q,False)))
                        if minus is not None:
                            rows.append(self.address[minus[0]]); cols.append(col); data.append(plus[1]*minus[1])
            # This backend currently uses Ms=0, hence S²=S_- S_+.
            self._spin_matrix = sparse.csr_array((data,(rows,cols)),shape=(self.dimension,self.dimension))
        return float(np.vdot(state,self._spin_matrix @ state).real)
