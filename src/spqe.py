"""Exact, spin-adapted SPQE screening and projective amplitude microiterations.

This project uses exact SPQE-style residuals within the single/double spatial
occupation neighborhood of HF and selected amplitudes. Selection promotes whole
scalar blocks.
"""
from itertools import combinations
from math import hypot
import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeResult

from excitations import apply_fermion_word, singlet_excitations_for_block
from excitation_neighborhood import ExcitationNeighborhood

def _real_residuals(residuals):
    if np.iscomplexobj(residuals) and np.any(abs(residuals.imag) > 1e-10):
        raise ValueError('PQE with real amplitudes requires real projected residuals')
    return np.asarray(residuals.real, dtype=float)


class ReferenceProjector:
    """Fixed signed CSF reference rows; storage scales with their nonzero terms."""

    def __init__(self, excitations, address, hf_index, n_modes, dimension):
        self.excitations = tuple(excitations)
        self.ids = tuple(e.id for e in self.excitations)
        rows, columns, data = [0], [], []
        block, components = None, {}
        for excitation in self.excitations:
            # Couplings of one spatial block share canonical monomials, with
            # different coefficients. Compute their determinant/phase once.
            if excitation.block != block:
                block, components = excitation.block, {}
            for term in excitation.terms:
                key = term.holes, term.particles
                if key not in components:
                    index, phase = apply_fermion_word(hf_index, n_modes, term.word)
                    components[key] = address[index], phase
                column, phase = components[key]
                columns.append(column); data.append(term.coefficient*phase)
            rows.append(len(data))
        # Keep the canonical term summation order rather than sorting columns.
        self.matrix = sparse.csr_array((np.asarray(data, dtype=float),
                                       np.asarray(columns, dtype=np.intp),
                                       np.asarray(rows, dtype=np.intp)),
                                      shape=(len(self.ids), dimension))

    def project(self, vector):
        return _real_residuals(self.matrix @ vector)


def project_residual(excitations, vector, address, hf_index, n_modes):
    """One-off <Phi_mu|vector>; iterative callers reuse ReferenceProjector."""
    return ReferenceProjector(excitations, address, hf_index, n_modes, len(vector)).project(vector)


class SPQESelection(ExcitationNeighborhood):
    """Residual selection within the single/double neighborhood of selected CSFs.

    HF is always a source. Spatial occupation distance is half the occupation
    L1 distance, so whole scalar blocks are eligible within two electron moves
    of any source block. Only eligible blocks with nonzero determinant support
    get a scalar basis. Values are recomputed on every call.
    """
    def __init__(self, indices, hf_index, n_modes, constraints, address=None):
        self.indices = indices
        self.address = ({int(index): i for i, index in enumerate(indices)}
                        if address is None else address)
        super().__init__(hf_index, n_modes, constraints)
        self._support_rows = {}
        self.blocks = []
        self._projector = None
        self._active_ids = None

    def _support(self, block):
        cached = self._support_rows.get(block)
        if cached is None:
            occupations = self._occupations(block)
            opens = [i for i, n in enumerate(occupations) if n == 1]
            index = sum(3 << (self.n_modes-2*i-2) for i, n in enumerate(occupations) if n == 2)
            index |= sum(1 << (self.n_modes-2*i-2) for i in opens)  # beta spins
            rows = []
            for alpha in combinations(opens, len(opens)//2):
                flipped = sum(3 << (self.n_modes-2*i-2) for i in alpha)
                rows.append(self.address[index ^ flipped])
            cached = self._support_rows[block] = np.asarray(rows, dtype=np.intp)
        return cached

    def _discover(self, vector, excitations):
        self._update_candidates(excitations)
        for block in sorted(self._candidates-self._blocks.keys()):
            if np.any(vector[self._support(block)] != 0):
                self._blocks[block] = singlet_excitations_for_block(*block)
        blocks = sorted(self._candidates & self._blocks.keys())
        if self._projector is None or blocks != self.blocks:
            self.blocks = blocks
            generators, block_rows = [], []
            for i, block in enumerate(self.blocks):
                generators.extend(self._blocks[block])
                block_rows.extend([i]*len(self._blocks[block]))
            self._block_rows = np.asarray(block_rows, dtype=np.intp)
            self._projector = ReferenceProjector(generators, self.address, self.hf_index,
                                                 self.n_modes, len(self.indices))
            self._active_ids = None

    def select(self, vector, excitations, epsilon):
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError('Selection threshold must be finite and nonnegative')
        self._discover(vector, excitations)
        active_ids = tuple(e.id for e in excitations)
        if self._active_ids != active_ids:
            present = set(active_ids)
            self._missing = np.array([identifier not in present for identifier in self._projector.ids], dtype=bool)
            self._active_ids = active_ids
        residuals = self._projector.project(vector)
        residuals[~self._missing] = 0.
        norms = np.sqrt(np.bincount(self._block_rows, weights=residuals**2,
                                   minlength=len(self.blocks)))
        scores = [(self.blocks[i], float(norms[i]), 'projective_residual')
                  for i in np.flatnonzero(norms)]
        proposals, diagnostics = _select_scores(scores, epsilon)
        diagnostics.update(candidate_blocks=len(self._candidates),
                           source_blocks=len(self._sources))
        return proposals, diagnostics


def select_blocks(vector, indices, hf_index, n_modes, constraints, excitations, epsilon):
    """One-off screening; the driver reuses SPQESelection across macroiterations."""
    return SPQESelection(indices, hf_index, n_modes, constraints).select(vector, excitations, epsilon)


def _select_scores(scores, epsilon):
    # Accumulate discarded contributions from smallest to largest. Avoid
    # subtracting nearly equal squared norms near a tight selection threshold.
    discarded = 0.
    selected = []
    for proposal in sorted(scores, key=lambda item: (item[1], item[0])):
        candidate = hypot(discarded, proposal[1])
        if candidate <= epsilon:
            discarded = candidate
        else:
            selected.append(proposal)
    selected.sort(key=lambda item: (-item[1], item[0]))
    return selected, dict(candidate_residual_norm=float(np.linalg.norm([s for _, s, _ in scores])),
                          excluded_residual_norm=float(discarded))


def solve_pqe(objective, initial, gaps, *, residual_tolerance=1e-6, maxiter=300, memory=8):
    """Gap-preconditioned quasi-Newton/DIIS with residual-norm backtracking.

    objective returns (energy, selected residuals). Neither energy gradients nor
    energy decrease enter the update or convergence test. Failed solves are
    returned to the caller without committing their trial amplitudes.
    """
    x = np.array(initial, dtype=float, copy=True)
    gaps = np.asarray(gaps, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError('Expected a finite PQE parameter vector')
    if gaps.shape != x.shape or np.any(gaps <= 0) or not np.all(np.isfinite(gaps)):
        raise ValueError('Expected positive finite PQE denominator magnitudes')
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0 or type(maxiter) is not int or maxiter < 1:
        raise ValueError('Expected positive residual tolerance and iteration limit')
    points, errors = [], []
    energy, residual = objective(x)
    for iteration in range(maxiter+1):
        norm = np.linalg.norm(residual)
        if not np.isfinite(energy) or not np.all(np.isfinite(residual)):
            message = 'Nonfinite projective energy or residual'
            break
        if norm <= residual_tolerance:
            return OptimizeResult(x=x, fun=energy, residual=residual, success=True,
                                  nit=iteration, message='L2 projected residual converged')
        if iteration == maxiter:
            message = 'PQE iteration limit reached'
            break
        step = -residual/gaps
        points.append(x+step)
        errors.append(residual/gaps)
        points, errors = points[-memory:], errors[-memory:]
        directions = []
        if iteration >= 2:
            gram = np.array(errors) @ np.array(errors).T
            scale = np.max(abs(gram))
            system = np.ones((len(gram)+1, len(gram)+1))
            system[:-1, :-1] = gram/scale if scale else gram
            system[-1, -1] = 0.
            rhs = np.zeros(len(system)); rhs[-1] = 1.
            weights = np.linalg.lstsq(system, rhs, rcond=1e-12)[0][:-1]
            if np.max(abs(weights)) <= 20:
                directions.append(weights @ np.array(points)-x)
        directions.append(step)
        accepted = False
        for direction in directions:
            alpha = 1.
            for _ in range(20):
                trial = x+alpha*direction
                trial_energy, trial_residual = objective(trial)
                if (np.isfinite(trial_energy) and np.all(np.isfinite(trial_residual))
                        and np.linalg.norm(trial_residual) <= (1-1e-4*alpha)*norm):
                    x, energy, residual = trial, trial_energy, trial_residual
                    accepted = True
                    break
                alpha *= .5
            if accepted:
                break
        if not accepted:
            message = 'PQE residual line search failed'
            break
    return OptimizeResult(x=x, fun=energy, residual=residual, success=False,
                          nit=iteration, message=message)
