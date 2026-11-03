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
from solvers import davidson_response


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
    omissions could change promotion or the reported external maximum.
    """

    def __init__(self, candidates, threshold, amplitude_scale, *, approximation='dressed'):
        self.candidates = candidates
        self.approximation = approximation
        self.threshold = threshold
        self.cutoff = threshold/(2*amplitude_scale) if amplitude_scale else np.inf
        if threshold == 0:
            self.cutoff = 0.
        self.columns = {}
        self.indices = np.empty(0, dtype=int)
        self.matrix = np.empty((0, 0))
        self.refined_columns = 0

    def _columns(self, indices):
        if self.approximation == 'dressed':
            return self.candidates.matrix_columns(indices)
        return self.candidates.matrix_columns(indices, approximation=self.approximation)

    def _store(self, index, values, external, cutoff):
        magnitude = abs(values)
        keep = (magnitude >= cutoff) & external & (magnitude > 0)
        # Keep an external maximizer even when all couplings are subthreshold.
        # This usually resolves the exact diagnostic maximum without refinement.
        strongest = np.argmax(np.where(external, magnitude, 0.))
        if external[strongest] and magnitude[strongest]:
            keep[strongest] = True
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
        # Approximate external couplings must never enter the selected LR
        # matrix. Measure only its selected rows in a restricted exact context.
        exact = self.candidates._restrict(indices) if self.approximation != 'dressed' else None
        batch_size = self.candidates._column_batch_size()
        for start in range(0, len(missing), batch_size):
            chosen = missing[start:start+batch_size]
            values = self._columns(chosen)
            positions = np.searchsorted(indices, chosen)
            internal = values[indices] if exact is None else USCCResponse.matrix_columns(exact, positions)
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
        while True:
            passed = external.copy() if self.threshold == 0 else np.zeros_like(external)
            maximum = 0.
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
                maximum = max(maximum, float(np.max(values, initial=0.)))
            unresolved = bounds > maximum
            if self.threshold:
                unresolved |= bounds >= self.threshold
            positions = np.flatnonzero(unresolved)
            if not len(positions):
                return passed, maximum
            # Only unresolved omissions trigger another screening-column pass.
            positions = positions[np.argsort(bounds[positions])[-self.candidates._column_batch_size():]]
            chosen = indices[positions]
            values = self._columns(chosen)
            target = min(self.threshold, maximum)/2
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
    dimension = len(excitations)
    sources = np.array([candidates.property_vector(op) for op in properties])
    blocks = {}
    for i,excitation in enumerate(excitations):
        blocks.setdefault(excitation.block,[]).append(i)
    def promote(selected,scores,threshold):
        promoted = set(selected)
        for indices in blocks.values():
            if np.max(scores[indices]) >= threshold:
                promoted.update(indices)
        return promoted
    selected = {i for i,e in enumerate(excitations) if e.rank <= seed_rank}
    selected = promote(selected,np.max(abs(sources),axis=0),property_threshold)
    history = []
    amplitude_scale = max(np.linalg.norm(source) for source in sources)/denominator_floor
    coupling_columns = _CouplingColumns(candidates, response_threshold, amplitude_scale,
                                        approximation=COUPLING_MODES[response_score])
    diagonal = candidates.orbital_diagonal()
    converged = False
    for cycle in range(max_cycles+1):
        P = np.array(sorted(selected),dtype=int)
        weights = np.zeros(len(P))
        max_internal_norm = 0.
        coupling_columns.add(P)
        internal = coupling_columns.matrix
        inner_history = None
        for shift in branches:
            dtype = complex if shift.imag else float
            shift = shift if shift.imag else shift.real
            def matvec(vector):
                return internal @ vector
            for source in sources:
                rhs = source[P].astype(dtype)
                if not np.any(rhs):
                    solution = np.zeros_like(rhs)
                else:
                    solution,inner_history = davidson_response(matvec,rhs,diagonal[P],history=inner_history,
                        omega=-complex(shift).real,gamma=-complex(shift).imag,tol=solver_tolerance*np.linalg.norm(rhs),
                        denominator_floor=denominator_floor)
                residual = rhs-internal @ solution+shift*solution
                if np.linalg.norm(residual) > solver_tolerance*np.linalg.norm(rhs)*1.01 and np.any(rhs):
                    raise RuntimeError('Inner selected response solve did not converge (true residual)')
                max_internal_norm = max(max_internal_norm,float(np.linalg.norm(residual)))
                np.maximum(weights, abs(solution), out=weights)
        passed, maximum = coupling_columns.screen(weights, P)
        grown = promote(selected,passed,.5)
        added = grown-selected
        history.append(dict(cycle=cycle,dimension=len(P),ranks={str(r):sum(excitations[i].rank == r for i in P) for r in sorted({e.rank for e in excitations})},max_internal_residual_norm=max_internal_norm,max_external_score=maximum,added=[excitations[i].id for i in sorted(added)]))
        history[-1].update(coupling_cache_bytes=coupling_columns.stored_bytes,
                           coupling_dense_columns=coupling_columns.dense_columns,
                           coupling_refined_columns=coupling_columns.refined_columns)
        if not added:
            converged = True
            break
        if cycle == max_cycles:
            break
        selected = grown
    indices = tuple(sorted(selected))
    response = candidates._restrict(indices, matrix=internal)
    response._davidson_history = inner_history
    return ResponseSelection(response,indices,history,converged,'threshold_converged' if converged else 'cycle_limit',property_threshold,response_threshold,response_score)
