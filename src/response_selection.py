"""Finite-candidate +V/+X selection for fixed-ground-state USCC response.

property_threshold compares |b_a| (property units); response_threshold compares
|b_a-(Kx)_a|/max(|K_aa-z|, denominator_floor) (property/energy units), or the
coupling score max_i |K_ai*x_i| (property units). Scores
take the maximum over properties and both frequency branches. Entire spatial
spin-coupling blocks within the supplied pool are promoted. Convergence refers
to that finite pool.
"""
from dataclasses import dataclass
import numpy as np
from uscc_response import USCCResponse
from solvers import davidson_response


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


def _residual_scores(candidates, external, residuals, shifts, threshold, floor):
    """Exact promotion/max score with lazy diagonals and rigorous upper bounds.

    Unmeasured rejected entries may retain upper bounds internally. They are
    strictly below the threshold and no larger than the measured maximum, so
    neither the selected set nor its reported maximum score changes.
    """
    scores = np.zeros(len(candidates.space))
    if not len(external):
        return scores
    residuals = np.asarray(residuals)[:, external]
    shifts = np.asarray(shifts)
    upper = np.max(residuals/np.maximum(abs(shifts.imag), floor)[:, None], axis=0)
    values = upper.copy()
    measured = np.zeros(len(external), dtype=bool)

    def evaluate(positions):
        diagonal = candidates.diagonal(external[positions])
        values[positions] = np.max(residuals[:, positions]/np.maximum(abs(diagonal[None, :]-shifts[:, None]), floor), axis=0)
        measured[positions] = True

    # Scores that can reach the threshold require exact denominators. Reuse
    # previously measured entries even if their current upper bound is small.
    cached = np.zeros(len(external), dtype=bool) if candidates._diagonal is None else np.isfinite(candidates._diagonal[external])
    positions = np.flatnonzero((upper >= threshold) | cached)
    if len(positions):
        evaluate(positions)
    best = float(np.max(values[measured])) if np.any(measured) else 0.
    while True:
        remaining = np.flatnonzero(~measured & (upper > best))
        if not len(remaining):
            break
        # Batch the largest bounds while resolving the exact diagnostic max.
        chosen = remaining[np.argsort(upper[remaining])[-32:]]
        evaluate(chosen)
        best = max(best, float(np.max(values[chosen])))
    scores[external] = values
    return scores


def select_response(candidates, properties, *, property_threshold=1e-4,
                    response_threshold=1e-4, response_score='residual', seed_rank=1, shifts=(0j,),
                    denominator_floor=.05, solver_tolerance=1e-11, max_cycles=50):
    """Select from a response object's independent rank-limited candidate pool.

    U and its ground-state parameters remain fixed. A zero threshold includes
    the entire supplied pool. ``seed_rank=0`` allows property-only seeding.
    ``shifts`` contains omega+i*gamma; both z and -z are solved automatically.
    Coupling scoring caches columns of K and ignores cancellations. Its units
    match +V, but it is a heuristic, not an external-equation error criterion.
    """
    if any(not np.isfinite(x) or x < 0 for x in (property_threshold,response_threshold)):
        raise ValueError('Selection thresholds must be finite and nonnegative')
    if response_score not in ('residual','coupling'):
        raise ValueError('response_score must be residual or coupling')
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
    diagonal = candidates.orbital_diagonal() if response_score == 'residual' else np.zeros(dimension)
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
    column_cache = {}
    converged = False
    for cycle in range(max_cycles+1):
        P = np.array(sorted(selected),dtype=int)
        Q = np.array(sorted(set(range(dimension))-selected),dtype=int)
        scores = np.zeros(dimension)
        residuals, residual_shifts = [], []
        max_external_norm, max_internal_norm = 0., 0.
        if response_score == 'coupling':
            missing = [i for i in P if i not in column_cache]
            if missing:
                measured = candidates.matrix_columns(missing)
                column_cache.update(zip(missing, measured.T))
            columns = np.column_stack([column_cache[i] for i in P]) if len(P) else np.empty((dimension,0))
            diagonal[P] = np.diag(columns[P]).real
            internal = columns[P]
        inner_history = None
        for shift in branches:
            dtype = complex if shift.imag else float
            shift = shift if shift.imag else shift.real
            def lift(vector):
                full = np.zeros(dimension,dtype=np.result_type(vector,dtype))
                full[P] = vector
                return full
            def matvec(vector):
                if response_score == 'coupling':
                    return internal @ vector
                return candidates.matvec(lift(vector))[P]
            for source in sources:
                rhs = source[P].astype(dtype)
                if not np.any(rhs):
                    solution = np.zeros_like(rhs)
                else:
                    solution,inner_history = davidson_response(matvec,rhs,diagonal[P],history=inner_history,
                        omega=-complex(shift).real,gamma=-complex(shift).imag,tol=solver_tolerance*np.linalg.norm(rhs),
                        denominator_floor=denominator_floor)
                full = lift(solution)
                applied = columns @ solution if response_score == 'coupling' else candidates.matvec(full)
                residual = source-applied+shift*full
                if np.linalg.norm(residual[P]) > solver_tolerance*np.linalg.norm(rhs)*1.01 and np.any(rhs):
                    raise RuntimeError('Inner selected response solve did not converge (true residual)')
                max_internal_norm = max(max_internal_norm,float(np.linalg.norm(residual[P])))
                max_external_norm = max(max_external_norm,float(np.linalg.norm(residual[Q])))
                if response_score == 'coupling':
                    contribution = np.max(abs(columns)*abs(solution)[None,:],axis=1) if len(P) else np.zeros(dimension)
                else:
                    residuals.append(abs(residual))
                    residual_shifts.append(shift)
                    continue
                scores = np.maximum(scores,contribution)
        if response_score == 'residual':
            scores = _residual_scores(candidates, Q, residuals, residual_shifts, response_threshold, denominator_floor)
        grown = promote(selected,scores,response_threshold)
        added = grown-selected
        history.append(dict(cycle=cycle,dimension=len(P),ranks={str(r):sum(excitations[i].rank == r for i in P) for r in sorted({e.rank for e in excitations})},max_external_residual_norm=max_external_norm,max_internal_residual_norm=max_internal_norm,max_external_score=float(np.max(scores[Q])) if len(Q) else 0.,added=[excitations[i].id for i in sorted(added)]))
        if not added:
            converged = True
            break
        if cycle == max_cycles:
            break
        selected = grown
    indices = tuple(sorted(selected))
    if response_score == 'coupling':
        response = candidates._restrict(indices, matrix=columns[P])
    else:
        response = candidates if len(indices) == dimension else candidates._restrict(indices)
    response._davidson_history = inner_history
    return ResponseSelection(response,indices,history,converged,'threshold_converged' if converged else 'cycle_limit',property_threshold,response_threshold,response_score)
