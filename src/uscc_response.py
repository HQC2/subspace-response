"""Fixed-space state-transfer qLR on a correlated USCC reference.

Reference CSFs are independent of the selected ground-state generators. The
active-space equations have A=K, B=0 and an identity response metric.
They also match self-consistent qLR when its residual corrections vanish.
"""
from collections import OrderedDict
from copy import copy, deepcopy
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh
from pyscf import lib
from pyscf.fci import cistring, direct_spin1

from excitations import singlet_excitations
from uscc_simulator import SectorSimulator
from solvers import davidson_response


class ResponseSpace:
    """Orthonormal excited reference CSFs stored only in the electron sector."""

    def __init__(self, simulator, excitations):
        self.excitations = tuple(excitations)
        self.ids = tuple(e.id for e in self.excitations)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('Duplicate response excitation IDs')
        hf_index = int(simulator.indices[np.flatnonzero(simulator.initial)[0]])
        rows, cols, data = [], [], []
        for col,excitation in enumerate(self.excitations):
            for term in excitation.terms:
                if any(h >= simulator.integrals.electrons for h in term.holes) or any(p < simulator.integrals.electrons or p >= simulator.n_modes for p in term.particles):
                    raise ValueError('Response excitation is incompatible with the active reference')
                result = term.apply(hf_index,simulator.n_modes)
                if result is None or result[0] not in simulator.address:
                    raise ValueError('Response excitation leaves the active spin/electron sector')
                rows.append(simulator.address[result[0]]); cols.append(col); data.append(result[1])
        self.columns = sparse.csc_array((data,(rows,cols)),shape=(simulator.dimension,len(self.ids)))
        gram_error = self.columns.T @ self.columns-sparse.eye_array(len(self.ids),format='csc')
        if gram_error.nnz and np.max(abs(gram_error.data)) > 1e-10:
            raise ValueError('Response reference states must be orthonormal')

    def __len__(self):
        return len(self.excitations)


class USCCResponse:
    """State-transfer response with fixed ground state and independent reference CSFs.

    The saved ground-state unitary defines U†HU and U†OU throughout both
    response operator selection and the final state-transfer LR calculation.
    Property inputs are real symmetric active-space spatial-orbital integrals;
    returned property vectors are transition couplings, not gate derivatives.
    """

    def __init__(self, wavefunction, excitations=None, *, max_rank=2, sector_threads=None):
        self.integrals = wavefunction.integrals
        self.metadata = deepcopy(wavefunction.metadata)
        self.theta = np.array(wavefunction.theta,dtype=float,copy=True)
        self.theta.flags.writeable = False
        self.ground_excitations = tuple(wavefunction.excitations)
        self.trotter_steps = wavefunction.trotter_steps
        self.spin_adaptation = wavefunction.spin_adaptation
        self.trotter_order = wavefunction.trotter_order
        threads = wavefunction.sector_threads if sector_threads is None else sector_threads
        self.sector_kernels = wavefunction.sector_kernels
        self.rotation_threads = wavefunction.rotation_threads
        self.simulator = SectorSimulator(self.integrals,sector_threads=threads,kernels=self.sector_kernels,rotation_threads=self.rotation_threads, spin_adaptation=self.spin_adaptation, trotter_order=self.trotter_order)
        if excitations is None:
            if type(max_rank) is not int or max_rank < 1:
                raise ValueError('max_rank must be a positive integer')
            excitations = singlet_excitations(self.integrals.electrons,self.integrals.qubits,range(1,max_rank+1))
        self.space = ResponseSpace(self.simulator,excitations)
        self.ground_state = self._unitary(self.simulator.initial)
        applied = self.simulator.apply_hamiltonian(self.ground_state)
        self.energy = float(np.vdot(self.ground_state,applied).real)
        self.ground_residual_norm = float(np.linalg.norm(applied-self.energy*self.ground_state))
        self.ground_state.flags.writeable = False
        self._orbital_diagonal = None
        self._matrix = None
        self._solution_cache = OrderedDict()
        self._davidson_history = None
        self._property_cache = {}
        self._one_body_links = None

    def _restrict(self, indices, *, matrix=None):
        """Reuse a frozen ground-state context and exact projected data on P.

        Selection supplies K_PP when it has already measured its columns. The
        simulator/rotation plan is shared; response-space caches are independent.
        As with a single simulator, concurrent calls require external locking.
        """
        indices = np.asarray(indices, dtype=int)
        result = copy(self)
        result.metadata = deepcopy(self.metadata)
        result.space = copy(self.space)
        result.space.excitations = tuple(self.space.excitations[i] for i in indices)
        result.space.ids = tuple(self.space.ids[i] for i in indices)
        result.space.columns = self.space.columns[:, indices]
        result._property_cache = {key: value[indices].copy() for key, value in self._property_cache.items()}
        result._orbital_diagonal = None if self._orbital_diagonal is None else self._orbital_diagonal[indices].copy()
        result._solution_cache = OrderedDict()
        result._davidson_history = None
        if matrix is None and self._matrix is not None:
            matrix = self._matrix[np.ix_(indices, indices)]
        result._matrix = None if matrix is None else np.array(matrix, copy=True)
        if result._matrix is not None:
            result._matrix.flags.writeable = False
        return result

    def _unitary(self, vector, *, adjoint=False):
        return self.simulator.apply_unitary(vector,self.theta,self.ground_excitations,self.trotter_steps,adjoint=adjoint)

    def matvec(self, vector):
        vector = np.asarray(vector)
        if vector.shape != (len(self.space),) or not np.all(np.isfinite(vector)):
            raise ValueError('Expected a finite vector matching the response space')
        if self._matrix is not None:
            return self._matrix @ vector
        state = self._unitary(self.space.columns @ vector)
        applied = self._unitary(self.simulator.apply_hamiltonian(state),adjoint=True)
        return self.space.columns.conj().T @ applied-self.energy*vector

    def matrix(self):
        """Explicit small-system oracle; iterative solvers use matvec instead."""
        if self._matrix is None:
            dimension = len(self.space)
            matrix = np.empty((dimension, dimension))
            unit = np.zeros(dimension)
            for i in range(dimension):
                unit[i] = 1.
                matrix[:, i] = self.matvec(unit)
                unit[i] = 0.
            matrix.flags.writeable = False
            self._matrix = matrix
        return self._matrix.copy()

    def matrix_columns(self, indices, *, batch_size=None, approximation='dressed'):
        """Candidate K columns, optionally approximated for +X screening only.

        ``dressed`` uses U†HU; ``bare`` uses H; ``linear`` uses H+[H,A], with
        A=sum(theta_mu*(T_mu-T_mu†)). All subtract the fixed reference energy.
        Approximate columns never populate the exact matrix cache. Each
        sector-state buffer is bounded by the batch policy; no dense dressed
        basis or commutator matrix is constructed.
        """
        if approximation not in ('dressed', 'bare', 'linear'):
            raise ValueError('Coupling approximation must be dressed, bare or linear')
        indices = self._column_indices(indices)
        if batch_size is None:
            batch_size = self._column_batch_size()
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if approximation == 'dressed' and self._matrix is not None:
            return self._matrix[:, indices].copy()
        result = np.empty((len(self.space), len(indices)))
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start+batch_size]
            states = self.space.columns[:, chosen].T.toarray()
            if approximation == 'dressed':
                states = self.simulator.apply_unitary_batch(states, self.theta, self.ground_excitations, self.trotter_steps)
                applied = self.simulator.apply_hamiltonian_batch(states)
                applied = self.simulator.apply_unitary_batch(applied, self.theta, self.ground_excitations, self.trotter_steps, adjoint=True)
            else:
                applied = self.simulator.apply_hamiltonian_batch(states)
                if approximation == 'linear':
                    # e^-A H e^A = H + H A - A H + O(theta**2).
                    applied = (applied
                               + self.simulator.apply_hamiltonian_batch(self._selection_generator_action(states))
                               - self._selection_generator_action(applied))
            result[:, start:start+len(chosen)] = self.space.columns.conj().T @ applied.T
            result[chosen, start+np.arange(len(chosen))] -= self.energy
        return result

    def _selection_generator_action(self, states):
        """Apply A to independent row states without assembling a sector matrix."""
        coefficients = {}
        for excitation, theta in zip(self.ground_excitations, self.theta):
            for term in excitation.terms:
                key = term.holes, term.particles
                coefficients[key] = coefficients.get(key, 0.)+theta*term.coefficient
        result = np.zeros_like(states)
        for (holes, particles), coefficient in coefficients.items():
            if coefficient:
                source, target, signs = self.simulator.pairs(holes, particles)
                result[:, target] += coefficient*signs*states[:, source]
                result[:, source] -= coefficient*signs*states[:, target]
        return result

    def _column_indices(self, indices):
        raw = np.asarray(indices)
        if raw.ndim != 1 or (len(raw) and (raw.dtype.kind not in 'iu' or np.any(raw < 0) or np.any(raw >= len(self.space)))):
            raise ValueError('Expected response column indices')
        return raw.astype(int)

    def _column_batch_size(self):
        limit = min(32, max(1, 64*1024**2//(8*self.simulator.dimension)))
        workers = self.simulator.sector_threads or lib.num_threads()
        return limit-limit%workers if limit >= workers else limit

    def orbital_diagonal(self):
        """Cheap Fock excitation gaps, used only to precondition linear solves.

        These are not dressed matrix elements and never replace the response
        operator. Coupling selection has no energy-denominator score.
        """
        if self._orbital_diagonal is None:
            self._orbital_diagonal = self.integrals.excitation_gaps(self.space.excitations)
        return self._orbital_diagonal.copy()

    def property_vector(self, one_body):
        """b_mu=<mu|U† O U|HF> for real symmetric spin-summed one-body O."""
        one_body = np.asarray(one_body)
        norb = self.simulator.n_orbitals
        if one_body.shape != (norb,norb) or np.iscomplexobj(one_body) or not np.all(np.isfinite(one_body)) or not np.allclose(one_body,one_body.T,rtol=0,atol=1e-12):
            raise ValueError('Expected real symmetric active-space one-electron integrals')
        one_body = np.ascontiguousarray(one_body,dtype=float)
        key = one_body.tobytes()
        if key not in self._property_cache:
            sim = self.simulator
            if self._one_body_links is None:
                links = cistring.gen_linkstr_index_trilidx(range(norb),sim.n_alpha)
                self._one_body_links = (links,links)
            ci = np.ascontiguousarray((sim.phases*self.ground_state).reshape(sim.shape))
            with lib.with_omp_threads(sim.sector_threads):
                applied = np.asarray(direct_spin1.contract_1e(one_body,ci,norb,(sim.n_alpha,sim.n_beta),link_index=self._one_body_links)).ravel()
            applied = self._unitary(sim.phases*applied,adjoint=True)
            self._property_cache[key] = self.space.columns.conj().T @ applied
        return self._property_cache[key].copy()

    def dipole_integrals(self):
        """Electronic dipole (-r) integrals in the saved active MOs (a.u.).

        Nuclear and frozen-core constants have zero excited-reference coupling.
        """
        import pyscf
        meta = self.metadata
        required = ('symbols','geometry','basis','unit','mo_coeff','inactive_orbitals')
        if any(key not in meta for key in required):
            raise ValueError('Saved molecular orbital metadata is required for dipole integrals')
        molecule = pyscf.M(atom=list(zip(meta['symbols'],meta['geometry'])),basis=meta['basis'],unit=meta['unit'],charge=meta.get('charge',0),spin=0,verbose=0)
        core = meta['inactive_orbitals']
        coefficients = np.asarray(meta['mo_coeff'])[:,core:core+self.simulator.n_orbitals]
        return -np.einsum('pi,xpq,qj->xij',coefficients,molecule.intor('int1e_r'),coefficients)

    def solve(self, source, *, omega=0., gamma=0., tolerance=1e-9, maxiter=500):
        """Solve (K-(omega+i*gamma)I)x=source; audit the true residual.

        HF Fock excitation gaps precondition every response solve.
        Identical solves reuse a bounded cache on this frozen response context.
        """
        source = np.asarray(source)
        if source.shape != (len(self.space),) or not np.all(np.isfinite(source)):
            raise ValueError('Expected a finite source matching the response space')
        if not np.isfinite(omega) or not np.isfinite(gamma) or not np.isfinite(tolerance) or tolerance <= 0 or type(maxiter) is not int or maxiter < 1:
            raise ValueError('Expected finite shifts and positive solver tolerances/limits')
        dtype = np.result_type(source,float,complex if gamma else float)
        source = source.astype(dtype,copy=False)
        if not np.any(source):
            return np.zeros_like(source)
        shift = omega+1j*gamma if gamma else omega
        key = (source.dtype.str, source.tobytes(), shift, tolerance, maxiter)
        if key in self._solution_cache:
            self._solution_cache.move_to_end(key)
            return self._solution_cache[key].copy()
        solution, history = davidson_response(self.matvec,source,self.orbital_diagonal(),history=self._davidson_history,
                                             omega=-omega,gamma=-gamma,tol=tolerance*np.linalg.norm(source),maxiter=maxiter)
        if not np.all(np.isfinite(solution)) or np.linalg.norm(self.matvec(solution)-shift*solution-source) > tolerance*np.linalg.norm(source)*1.01:
            raise RuntimeError('Response linear solve did not converge (true residual)')
        self._davidson_history = history
        self._solution_cache[key] = solution.copy()
        if len(self._solution_cache) > 16:
            self._solution_cache.popitem(last=False)
        return solution

    def linear_response(self, left, right=None, *, omega=0., gamma=0., **solver_options):
        """Physical two-branch susceptibility for H(F)=H-F*right.

        For electric dipoles at omega=gamma=0 this is the polarizability,
        2*b_left.T @ K^-1 @ b_right. Property matrices are real symmetric.
        """
        a = self.property_vector(left)
        b = a if right is None else self.property_vector(right)
        positive = self.solve(b,omega=omega,gamma=gamma,**solver_options)
        negative = positive if omega == 0 and gamma == 0 else self.solve(b,omega=-omega,gamma=-gamma,**solver_options)
        return np.real_if_close(np.vdot(a,positive+negative)).item()

    def eigenpairs(self, roots=5, *, tolerance=1e-9):
        dimension = len(self.space)
        if type(roots) is not int or roots < 1 or not dimension or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('Expected a nonempty response space and positive root count/tolerance')
        roots = min(roots,dimension)
        if dimension <= 32 or roots == dimension:
            energies,vectors = np.linalg.eigh(self.matrix())
            return energies[:roots],vectors[:,:roots]
        operator = LinearOperator((dimension,dimension),matvec=self.matvec,dtype=float)
        energies,vectors = eigsh(operator,k=roots,which='SA',tol=tolerance,v0=np.random.default_rng(29).normal(size=dimension))
        order = np.argsort(energies)
        return energies[order],vectors[:,order]
