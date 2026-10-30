"""Fixed-space stationary sc-qLR and its undressed (U=I) approximation.

Reference CSFs are independent of the selected ground-state generators. The
response model neglects residual-dependent A corrections and the B block.
Identity dressing uses the HF energy throughout, giving CIS/TDA for singles.
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
    """Response with fixed ground state and independently chosen reference CSFs.

    ``dressing='uscc'`` uses U†HU and U†OU. ``dressing='identity'`` uses the
    bare operators and HF energy, independent of the ground-state parameters.
    Property inputs are real symmetric active-space spatial-orbital integrals;
    returned property vectors are transition couplings, not gate derivatives.
    """

    def __init__(self, wavefunction, excitations=None, *, max_rank=2, dressing='uscc', sector_threads=None):
        if dressing not in ('uscc','identity'):
            raise ValueError('dressing must be uscc or identity')
        self.dressing = dressing
        self.integrals = wavefunction.integrals
        self.metadata = deepcopy(wavefunction.metadata)
        self.theta = np.array(wavefunction.theta,dtype=float,copy=True)
        self.theta.flags.writeable = False
        self.ground_excitations = tuple(wavefunction.excitations)
        self.trotter_steps = wavefunction.trotter_steps
        threads = wavefunction.sector_threads if sector_threads is None else sector_threads
        self.sector_kernels = getattr(wavefunction,'sector_kernels','auto')
        self.rotation_threads = getattr(wavefunction,'rotation_threads',None)
        self.simulator = SectorSimulator(self.integrals,sector_threads=threads,kernels=self.sector_kernels if dressing == 'uscc' else 'numpy',rotation_threads=self.rotation_threads)
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
        self._diagonal = None
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
        result._diagonal = None if self._diagonal is None else self._diagonal[indices].copy()
        result._orbital_diagonal = None if self._orbital_diagonal is None else self._orbital_diagonal[indices].copy()
        result._solution_cache = OrderedDict()
        result._davidson_history = None
        if matrix is None and self._matrix is not None:
            matrix = self._matrix[np.ix_(indices, indices)]
        result._matrix = None if matrix is None else np.array(matrix, copy=True)
        if result._matrix is not None:
            result._matrix.flags.writeable = False
            result._diagonal = result._matrix.diagonal().real.copy()
        return result

    def _unitary(self, vector, *, adjoint=False):
        if self.dressing == 'identity':
            return np.array(vector,copy=True)
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
            self._diagonal = matrix.diagonal().real.copy()
        return self._matrix.copy()

    def matrix_columns(self, indices, *, batch_size=None):
        """Exact candidate K columns with bounded batches of sector states.

        This caches no dense dressed basis. By default each sector-state buffer
        is at most 64 MiB (and at most 32 vectors), independent of pool size.
        """
        indices = self._column_indices(indices)
        if batch_size is None:
            batch_size = self._column_batch_size()
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if self._matrix is not None:
            return self._matrix[:, indices].copy()
        result = np.empty((len(self.space), len(indices)))
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start+batch_size]
            states = self.space.columns[:, chosen].T.toarray()
            if self.dressing == 'uscc':
                states = self.simulator.apply_unitary_batch(states, self.theta, self.ground_excitations, self.trotter_steps)
            applied = self.simulator.apply_hamiltonian_batch(states)
            if self.dressing == 'uscc':
                applied = self.simulator.apply_unitary_batch(applied, self.theta, self.ground_excitations, self.trotter_steps, adjoint=True)
            result[:, start:start+len(chosen)] = self.space.columns.conj().T @ applied.T
            result[chosen, start+np.arange(len(chosen))] -= self.energy
        if len(indices):
            if self._diagonal is None:
                self._diagonal = np.full(len(self.space), np.nan)
            self._diagonal[indices] = result[indices, np.arange(len(indices))]
        return result

    def diagonal(self, indices=None):
        """Exact dressed diagonal, optionally evaluating only requested entries."""
        if indices is None:
            indices = np.arange(len(self.space))
        else:
            indices = self._column_indices(indices)
        if self._diagonal is None:
            if self._matrix is not None:
                self._diagonal = self._matrix.diagonal().real.copy()
            else:
                self._diagonal = np.full(len(self.space), np.nan)
        missing = np.unique(indices[np.isnan(self._diagonal[indices])])
        batch_size = self._column_batch_size()
        for start in range(0, len(missing), batch_size):
            # <mu|U† H U|mu> needs only forward rotation sweeps.
            chosen = missing[start:start+batch_size]
            states = self.space.columns[:, chosen].T.toarray()
            if self.dressing == 'uscc':
                states = self.simulator.apply_unitary_batch(states, self.theta, self.ground_excitations, self.trotter_steps)
            applied = self.simulator.apply_hamiltonian_batch(states)
            for i, state, image in zip(chosen, states, applied):
                self._diagonal[i] = np.vdot(state, image).real-self.energy
        return self._diagonal[indices].copy()

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

        These are not dressed matrix elements and never replace the exact
        denominators in residual-based +X selection or the response operator.
        """
        if self._orbital_diagonal is None:
            occupied = range(self.integrals.electrons//2)
            eri = self.simulator.chemist_eri
            eps = np.diag(self.integrals.one_body).copy()
            for i in occupied:
                eps += 2*np.einsum('pp->p', eri[:, :, i, i])-np.einsum('pp->p', eri[:, i, i, :])
            self._orbital_diagonal = np.array([sum(eps[p] for p in e.spatial_particles)-sum(eps[h] for h in e.spatial_holes) for e in self.space.excitations])
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

    def solve(self, source, *, omega=0., gamma=0., tolerance=1e-9, maxiter=500, preconditioner='orbital'):
        """Solve (K-(omega+i*gamma)I)x=source; audit the true residual.

        Orbital gaps avoid constructing the entire exact dressed diagonal.
        ``preconditioner='exact'`` retains that option; ``'none'`` disables it.
        Identical solves reuse a bounded cache on this frozen response context.
        """
        source = np.asarray(source)
        if source.shape != (len(self.space),) or not np.all(np.isfinite(source)):
            raise ValueError('Expected a finite source matching the response space')
        if not np.isfinite(omega) or not np.isfinite(gamma) or not np.isfinite(tolerance) or tolerance <= 0 or type(maxiter) is not int or maxiter < 1:
            raise ValueError('Expected finite shifts and positive solver tolerances/limits')
        if preconditioner not in ('orbital','exact','none'):
            raise ValueError('preconditioner must be orbital, exact or none')
        dtype = np.result_type(source,float,complex if gamma else float)
        source = source.astype(dtype,copy=False)
        if not np.any(source):
            return np.zeros_like(source)
        shift = omega+1j*gamma if gamma else omega
        key = (source.dtype.str, source.tobytes(), shift, tolerance, maxiter, preconditioner)
        if key in self._solution_cache:
            self._solution_cache.move_to_end(key)
            return self._solution_cache[key].copy()
        diagonal = self.diagonal() if preconditioner == 'exact' else self.orbital_diagonal() if preconditioner == 'orbital' else None
        solution, history = davidson_response(self.matvec,source,diagonal,history=self._davidson_history,
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
