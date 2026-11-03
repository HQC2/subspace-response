"""Finite-candidate +V/+X selection for fixed-ground-state USCC response.

property_threshold compares |b_a| (property units); response_threshold compares
the coupling score max_i |K_ai*x_i| (property units). +X may use dressed,
bare or first-order commutator couplings; the response equations stay exact.
Scores take the maximum over properties and both frequency branches. Entire spatial
spin-coupling blocks within the supplied pool are promoted. Convergence refers
to that finite pool.
"""
from dataclasses import dataclass
import numpy as np
from uscc_response import USCCResponse


COUPLING_MODES = {'coupling': 'dressed', 'bare-coupling': 'bare', 'linear-coupling': 'linear'}


@dataclass
class ResponseSelection:
    response: USCCResponse
    indices: tuple
    history: list
    converged: bool
    stop_reason: str
    property_threshold: float
    response_threshold: float
    response_score: str


class _CouplingColumns:
    """Exact selected matrix and adaptively screened external couplings.

    Sparse columns retain a bound on every omitted element. Screening checks
    that bound with the current amplitudes and remeasures only columns whose
    omissions could change promotion.
    """

    def __init__(self, candidates, threshold, amplitude_scale):
        self.candidates = candidates
        self.threshold = threshold
        self.cutoff = threshold/(2*amplitude_scale) if amplitude_scale else np.inf
        if threshold == 0:
            self.cutoff = 0.
        self.columns = {}
        self.indices = np.empty(0, dtype=int)
        self.matrix = np.empty((0, 0))
        self.refined_columns = 0

    def _columns(self, indices):
        return self.candidates.matrix_columns(indices)

    def _store(self, index, values, external, cutoff):
        magnitude = abs(values)
        keep = (magnitude >= cutoff) & external & (magnitude > 0)
        rows = np.flatnonzero(keep)
        index_type = np.int32 if len(values) <= np.iinfo(np.int32).max else np.int64
        if len(rows)*(np.dtype(index_type).itemsize+values.dtype.itemsize) >= values.nbytes:
            data = values.copy()
            data[~external] = 0.
            self.columns[index] = (None, data, 0.)
        else:
            omitted = float(np.max(magnitude, where=external & ~keep, initial=0.))
            self.columns[index] = (rows.astype(index_type), values[rows], omitted)

    def add(self, indices):
        missing = np.array([i for i in indices if i not in self.columns], dtype=int)
        if not len(missing):
            return
        old_positions = np.searchsorted(indices, self.indices)
        matrix = np.empty((len(indices), len(indices)))
        matrix[np.ix_(old_positions, old_positions)] = self.matrix
        external = np.ones(len(self.candidates.space), dtype=bool)
        external[indices] = False
        batch_size = self.candidates._column_batch_size()
        for start in range(0, len(missing), batch_size):
            chosen = missing[start:start+batch_size]
            values = self._columns(chosen)
            positions = np.searchsorted(indices, chosen)
            internal = values[indices]
            matrix[:, positions] = internal
            # Hermiticity supplies new rows of previously measured columns.
            matrix[np.ix_(positions, old_positions)] = internal[old_positions].T
            for index, column in zip(chosen, values.T):
                self._store(index, column, external, self.cutoff)
        self.indices = indices.copy()
        self.matrix = matrix

    def screen(self, weights, indices):
        external = np.ones(len(self.candidates.space), dtype=bool)
        external[indices] = False
        if self.threshold == 0:
            return external
        while True:
            passed = np.zeros_like(external)
            bounds = np.empty(len(indices))
            for position, (index, weight) in enumerate(zip(indices, weights)):
                rows, data, omitted = self.columns[index]
                bounds[position] = omitted*weight
                if not weight:
                    continue
                if rows is None:
                    rows = np.flatnonzero(external)
                    values = abs(data[rows])*weight
                else:
                    active = external[rows]
                    rows, values = rows[active], abs(data[active])*weight
                passed[rows[values >= self.threshold]] = True
            positions = np.flatnonzero(bounds >= self.threshold)
            if not len(positions):
                return passed
            # Only unresolved omissions trigger another screening-column pass.
            positions = positions[np.argsort(bounds[positions])[-self.candidates._column_batch_size():]]
            chosen = indices[positions]
            values = self._columns(chosen)
            target = self.threshold/2
            for position, index, column in zip(positions, chosen, values.T):
                self._store(index, column, external, target/weights[position])
            self.refined_columns += len(chosen)

    @property
    def stored_bytes(self):
        return sum(data.nbytes+(0 if rows is None else rows.nbytes)
                   for rows, data, _ in self.columns.values())

    @property
    def dense_columns(self):
        return sum(rows is None for rows, _, _ in self.columns.values())


class _ConnectedCouplings:
    """Recompute weighted external scores by streaming connected CSF elements.

    The native path visits Slater--Condon singles/doubles and generator words.
    Contributions to a CSF are summed before taking an absolute value. It
    retains no selected/candidate columns. The portable oracle streams bounded
    batches through the existing operator actions, without a column cache.
    """

    def __init__(self, candidates, threshold, *, approximation):
        self.candidates, self.threshold, self.approximation = candidates, threshold, approximation
        self.native = None
        sim = candidates.simulator
        module = sim._native_module
        if module is not None and hasattr(module, 'ConnectedScreening') and sim.n_orbitals <= 31:
            coefficients = {}
            if approximation == 'linear':
                for excitation, theta in zip(candidates.ground_excitations, candidates.theta):
                    for term in excitation.terms:
                        key = term.holes, term.particles
                        coefficients[key] = coefficients.get(key, 0.)+theta*term.coefficient
            rules, amplitudes = [], []
            for (holes, particles), coefficient in coefficients.items():
                if not coefficient:
                    continue
                _, _, parity, constant = sim.excitation_layout(holes, particles)
                source = sum(1 << (sim.n_modes-h-1) for h in holes)
                target = sum(1 << (sim.n_modes-p-1) for p in particles)
                rules.append((source, source | target, parity, constant))
                amplitudes.append(coefficient)
            C = candidates.space.columns
            self.native = module.ConnectedScreening(
                np.ascontiguousarray(candidates.integrals.one_body), sim.chemist_eri,
                sim._strings, sim.indices, sim.phases,
                C.indptr, C.indices, C.data,
                np.asarray(rules, dtype=np.uint64).reshape(-1, 4), np.asarray(amplitudes, dtype=float))

    def screen(self, weights, indices):
        external = np.ones(len(self.candidates.space), dtype=bool)
        external[indices] = False
        if self.threshold == 0:
            return external
        if not np.any(external) or not np.any(weights):
            return np.zeros_like(external)
        if self.native is not None:
            sim = self.candidates.simulator
            from pyscf import lib
            return self.native.screen(np.ascontiguousarray(indices, dtype=np.int64),
                                      np.ascontiguousarray(weights), self.threshold,
                                      self.approximation == 'linear', sim.sector_threads or lib.num_threads())
        else:
            passed = np.zeros(len(self.candidates.space), dtype=bool)
            size = self.candidates._column_batch_size()
            for start in range(0, len(indices), size):
                chosen = indices[start:start+size]
                values = self.candidates.matrix_columns(chosen, approximation=self.approximation)
                passed |= np.any(abs(values)*weights[start:start+size] >= self.threshold, axis=1)
            return external & passed

    stored_bytes = dense_columns = refined_columns = 0


def select_response(candidates, properties, *, property_threshold=1e-4,
                    response_threshold=1e-4, response_score='coupling', seed_rank=1, shifts=(0j,),
                    denominator_floor=.05, solver_tolerance=1e-11, max_cycles=50):
    """Select from a response object's independent rank-limited candidate pool.

    U and its ground-state parameters remain fixed. A zero threshold includes
    the entire supplied pool. ``seed_rank=0`` allows property-only seeding.
    ``shifts`` contains omega+i*gamma; both z and -z are solved automatically.
    The +X modes are coupling (U†HU), bare-coupling (H), and linear-coupling
    (H+[H,A], A=sum(theta_mu*(T_mu-T_mu†))). Only external screening is
    approximated; +V, selected-space solves and final response use U†HU.
    Scores ignore cancellations and are not external-equation error criteria.
    """
    if any(not np.isfinite(x) or x < 0 for x in (property_threshold,response_threshold)):
        raise ValueError('Selection thresholds must be finite and nonnegative')
    if response_score not in COUPLING_MODES:
        raise ValueError('response_score must be '+', '.join(COUPLING_MODES))
    if type(seed_rank) is not int or seed_rank < 0 or type(max_cycles) is not int or max_cycles < 1:
        raise ValueError('Expected a nonnegative seed rank and positive cycle limit')
    if not np.isfinite(denominator_floor) or denominator_floor <= 0 or not np.isfinite(solver_tolerance) or solver_tolerance <= 0:
        raise ValueError('Denominator floor and solver tolerance must be positive')
    properties = list(properties)
    if not properties:
        raise ValueError('At least one property component is required')
    branches = []
    for shift in shifts:
        shift = complex(shift)
        if not np.isfinite(shift):
            raise ValueError('Frequency shifts must be finite')
        for branch in (shift,-shift):
            if branch not in branches:
                branches.append(branch)
    if not branches:
        raise ValueError('At least one frequency shift is required')
    excitations = candidates.space.excitations
    sources = np.array([candidates.property_vector(op) for op in properties])
    blocks = {}
    for i,excitation in enumerate(excitations):
        blocks.setdefault(excitation.block,[]).append(i)
    def promote(selected,passed):
        promoted = set(selected)
        for i in np.flatnonzero(passed):
            promoted.update(blocks[excitations[i].block])
        return promoted
    selected = {i for i,e in enumerate(excitations) if e.rank <= seed_rank}
    selected = promote(selected,np.any(abs(sources) >= property_threshold, axis=0))
    history = []
    approximation = COUPLING_MODES[response_score]
    if approximation == 'dressed':
        amplitude_scale = max(np.linalg.norm(source) for source in sources)/denominator_floor
        coupling_columns = _CouplingColumns(candidates, response_threshold, amplitude_scale)
    else:
        coupling_columns = _ConnectedCouplings(candidates, response_threshold, approximation=approximation)
    converged = False
    previous, previous_indices = None, None
    for cycle in range(max_cycles+1):
        P = np.array(sorted(selected),dtype=int)
        weights = np.zeros(len(P))
        max_internal_norm = 0.
        if approximation == 'dressed':
            coupling_columns.add(P)
            response = candidates._restrict(P, matrix=coupling_columns.matrix)
        else:
            response = candidates._restrict(P)
            response._retain_solution_images = True
            if previous is not None:
                response._seed_from_response(previous, np.searchsorted(P, previous_indices))
        for shift in branches:
            dtype = complex if shift.imag else float
            shift = shift if shift.imag else shift.real
            for source in sources:
                rhs = source[P].astype(dtype)
                solution, residual_norm = response._solve(rhs, omega=complex(shift).real,
                    gamma=complex(shift).imag, tolerance=solver_tolerance, denominator_floor=denominator_floor)
                max_internal_norm = max(max_internal_norm, residual_norm)
                np.maximum(weights, abs(solution), out=weights)
        passed = coupling_columns.screen(weights, P)
        grown = promote(selected,passed)
        added = grown-selected
        history.append(dict(cycle=cycle,dimension=len(P),ranks={str(r):sum(excitations[i].rank == r for i in P) for r in sorted({e.rank for e in excitations})},max_internal_residual_norm=max_internal_norm,added=[excitations[i].id for i in sorted(added)]))
        history[-1].update(coupling_cache_bytes=coupling_columns.stored_bytes,
                           coupling_dense_columns=coupling_columns.dense_columns,
                           coupling_refined_columns=coupling_columns.refined_columns)
        if not added:
            converged = True
            break
        if cycle == max_cycles:
            break
        selected = grown
        previous, previous_indices = response, P
    indices = tuple(sorted(selected))
    response._solution_images.clear()
    response._retain_solution_images = False
    return ResponseSelection(response,indices,history,converged,'threshold_converged' if converged else 'cycle_limit',property_threshold,response_threshold,response_score)
