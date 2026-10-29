"""Fixed-space stationary sc-qLR and its undressed (U=I) approximation.

Reference CSFs are independent of the selected ground-state generators. The
response model neglects residual-dependent A corrections and the B block.
Identity dressing uses the HF energy throughout, giving CIS/TDA for singles.
"""
from copy import deepcopy
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh, gmres
from pyscf import lib
from pyscf.fci import cistring, direct_spin1

from excitations import singlet_excitations
from uscc_simulator import SectorSimulator


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
        self.simulator = SectorSimulator(self.integrals,sector_threads=threads,use_numba=dressing == 'uscc')
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
        self._property_cache = {}
        self._one_body_links = None

    def _unitary(self, vector, *, adjoint=False):
        if self.dressing == 'identity':
            return np.array(vector,copy=True)
        return self.simulator.apply_unitary(vector,self.theta,self.ground_excitations,self.trotter_steps,adjoint=adjoint)

    def matvec(self, vector):
        vector = np.asarray(vector)
        if vector.shape != (len(self.space),) or not np.all(np.isfinite(vector)):
            raise ValueError('Expected a finite vector matching the response space')
        state = self._unitary(self.space.columns @ vector)
        applied = self._unitary(self.simulator.apply_hamiltonian(state),adjoint=True)
        return self.space.columns.conj().T @ applied-self.energy*vector

    def matrix(self):
        """Explicit small-system oracle; iterative solvers use matvec instead."""
        if not len(self.space):
            return np.empty((0,0))
        return np.column_stack([self.matvec(unit) for unit in np.eye(len(self.space))])

    def diagonal(self):
        if self._diagonal is None:
            self._diagonal = np.empty(len(self.space))
            for i in range(len(self.space)):
                unit = np.zeros(len(self.space)); unit[i] = 1.
                self._diagonal[i] = self.matvec(unit)[i].real
        return self._diagonal.copy()

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
        """Solve (K-(omega+i*gamma)I)x=source; nonconvergence raises an error."""
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
        diagonal = self.diagonal()-shift
        safe = np.where(abs(diagonal) > 1e-10,diagonal,1.)
        operator = LinearOperator((len(self.space),)*2,matvec=lambda v:self.matvec(v)-shift*v,dtype=dtype)
        preconditioner = LinearOperator(operator.shape,matvec=lambda v:v/safe,dtype=dtype)
        solution,info = gmres(operator,source,M=preconditioner,rtol=tolerance,atol=0.,maxiter=maxiter)
        if info or not np.all(np.isfinite(solution)) or np.linalg.norm(operator @ solution-source) > tolerance*np.linalg.norm(source)*1.01:
            raise RuntimeError(f'Response linear solve did not converge (info={info})')
        return solution

    def linear_response(self, left, right=None, *, omega=0., gamma=0., **solver_options):
        """Physical two-branch susceptibility for H(F)=H-F*right.

        For electric dipoles at omega=gamma=0 this is the polarizability,
        2*b_left.T @ K^-1 @ b_right. Property matrices are real symmetric.
        """
        a = self.property_vector(left)
        b = a if right is None else self.property_vector(right)
        positive = self.solve(b,omega=omega,gamma=gamma,**solver_options)
        negative = self.solve(b,omega=-omega,gamma=-gamma,**solver_options)
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
