"""Static response with bare excitation generators on a correlated USCC state.

This small-system diagnostic retains the real static energy Hessian, including
the terms represented by B in a doubled qLR formulation. It is distinct from
substituting U=I everywhere in the stationary projected sc-qLR equations.
"""
import numpy as np
from scipy import sparse
from pyscf import lib
from pyscf.fci import cistring, direct_spin1

from uscc_response import USCCResponse


class BareUSCCStaticResponse:
    """Local response of exp(sum x_mu*(T_mu-T_mu†))|Psi_USCC> at x=0.

    For real states/properties, let t_mu=(T_mu-T_mu†)|Psi> and
    r=(H-E)|Psi>. The half Hessian is

        K_mu,nu = t_mu.T (H-E) t_nu
                   - ((kappa_mu r).T t_nu + t_mu.T kappa_nu r)/2.

    The source is b_mu=t_mu.T O|Psi> and alpha=2*b_left.T K^-1 b_right.
    No orthonormal-tangent assumption is made. Null tangent directions are
    removed using their Gram matrix before solving. This is a static-only,
    dense diagnostic, not the projected K-model or its +V/+X selection API.
    A small ground residual is needed to interpret the local response as the
    susceptibility about a stationary ground state.
    """

    def __init__(self, wavefunction, excitations=None, *, max_rank=2,
                 sector_threads=None, metric_tolerance=1e-10):
        if not np.isfinite(metric_tolerance) or not 0 < metric_tolerance < 1:
            raise ValueError('metric_tolerance must be between zero and one')
        reference = USCCResponse(wavefunction,excitations,max_rank=max_rank,
                                 sector_threads=sector_threads)
        self.space = reference.space
        self.simulator = reference.simulator
        self.ground_state = reference.ground_state
        self.energy = reference.energy
        self.ground_residual_norm = reference.ground_residual_norm
        self.dipole_integrals = reference.dipole_integrals
        sim = self.simulator
        residual = sim.apply_hamiltonian(self.ground_state)-self.energy*self.ground_state
        tangents, rotated_residuals = [], []
        for excitation in self.space.excitations:
            generator = sparse.csr_array((sim.dimension,sim.dimension))
            for term in excitation.terms:
                source,target,sign = sim.pairs(term.holes,term.particles)
                tau = sparse.csr_array((term.coefficient*sign,(target,source)),
                                       shape=generator.shape)
                generator = generator+tau-tau.T
            tangents.append(generator @ self.ground_state)
            rotated_residuals.append(generator @ residual)
        n = len(self.space)
        self._tangents = np.column_stack(tangents) if n else np.empty((sim.dimension,0))
        kr = np.column_stack(rotated_residuals) if n else np.empty((sim.dimension,0))
        ht = np.column_stack([sim.apply_hamiltonian(t) for t in self._tangents.T]) if n else self._tangents.copy()
        self._metric = self._tangents.T @ self._tangents
        correction = kr.T @ self._tangents
        self._matrix = self._tangents.T @ (ht-self.energy*self._tangents)-(correction+correction.T)/2
        self._matrix = (self._matrix+self._matrix.T)/2
        self.ground_gradient_norm = float(np.linalg.norm(2*self._tangents.T @ residual))
        eigenvalues,eigenvectors = np.linalg.eigh(self._metric)
        keep = eigenvalues > metric_tolerance*max(1.,eigenvalues[-1] if n else 0.)
        self._orthogonalizer = eigenvectors[:,keep]/np.sqrt(eigenvalues[keep])
        self.effective_dimension = int(np.sum(keep))
        self._reduced_matrix = self._orthogonalizer.T @ self._matrix @ self._orthogonalizer
        self._property_cache = {}
        links = cistring.gen_linkstr_index_trilidx(range(sim.n_orbitals),sim.n_alpha)
        self._links = (links,links)

    def matrix(self):
        """Half the full static energy Hessian in bare-generator coordinates."""
        return self._matrix.copy()

    def metric(self):
        """Real tangent Gram matrix; generally not the identity on a USCC state."""
        return self._metric.copy()

    def property_vector(self, one_body):
        one_body = np.asarray(one_body)
        n = self.simulator.n_orbitals
        if one_body.shape != (n,n) or np.iscomplexobj(one_body) or not np.all(np.isfinite(one_body)) or not np.allclose(one_body,one_body.T,rtol=0,atol=1e-12):
            raise ValueError('Expected real symmetric active-space one-electron integrals')
        one_body = np.ascontiguousarray(one_body,dtype=float)
        key = one_body.tobytes()
        if key not in self._property_cache:
            sim = self.simulator
            ci = np.ascontiguousarray((sim.phases*self.ground_state).reshape(sim.shape))
            with lib.with_omp_threads(sim.sector_threads):
                applied = direct_spin1.contract_1e(one_body,ci,n,(sim.n_alpha,sim.n_beta),link_index=self._links)
            self._property_cache[key] = self._tangents.T @ (sim.phases*np.asarray(applied).ravel())
        return self._property_cache[key].copy()

    def solve(self, source):
        source = np.asarray(source)
        if source.shape != (len(self.space),) or not np.all(np.isfinite(source)):
            raise ValueError('Expected a finite source matching the response space')
        w = self._orthogonalizer
        return w @ np.linalg.solve(self._reduced_matrix,w.T @ source)

    def linear_response(self, left, right=None):
        """Static susceptibility; frequency-dependent response is not implemented."""
        a = self.property_vector(left)
        b = a if right is None else self.property_vector(right)
        return float(2*a @ self.solve(b))
