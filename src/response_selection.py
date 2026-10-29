"""Finite-candidate +V/+X selection for fixed-ground-state USCC response.

property_threshold compares |b_a| (property units); response_threshold compares
|b_a-(Kx)_a|/max(|K_aa-z|, denominator_floor) (property/energy units), or the
coupling score max_i |K_ai*x_i| (property units). Scores
take the maximum over properties and both frequency branches. Entire spatial
spin-coupling blocks within the supplied pool are promoted. Convergence refers
to that finite pool.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from uscc_response import USCCResponse


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
    diagonal = candidates.diagonal() if response_score == 'residual' else np.zeros(dimension)
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
        max_external_norm, max_internal_norm = 0., 0.
        if response_score == 'coupling':
            for i in P:
                if i not in column_cache:
                    unit = np.zeros(dimension); unit[i] = 1.
                    column_cache[i] = candidates.matvec(unit)
            columns = np.column_stack([column_cache[i] for i in P]) if len(P) else np.empty((dimension,0))
            diagonal[P] = np.diag(columns[P]).real
        for shift in branches:
            dtype = complex if shift.imag else float
            shift = shift if shift.imag else shift.real
            def lift(vector):
                full = np.zeros(dimension,dtype=np.result_type(vector,dtype))
                full[P] = vector
                return full
            def matvec(vector):
                if response_score == 'coupling':
                    return columns[P] @ vector-shift*vector
                return candidates.matvec(lift(vector))[P]-shift*vector
            safe = diagonal[P]-shift
            safe = np.where(abs(safe) > denominator_floor,safe,denominator_floor)
            operator = LinearOperator((len(P),)*2,matvec=matvec,dtype=dtype)
            preconditioner = LinearOperator(operator.shape,matvec=lambda v:v/safe,dtype=dtype)
            for source in sources:
                rhs = source[P].astype(dtype)
                if not np.any(rhs):
                    solution = np.zeros_like(rhs)
                else:
                    solution,info = gmres(operator,rhs,M=preconditioner,rtol=solver_tolerance,atol=0.,maxiter=500)
                    if info:
                        raise RuntimeError(f'Inner selected response solve did not converge (info={info})')
                full = lift(solution)
                applied = columns @ solution if response_score == 'coupling' else candidates.matvec(full)
                residual = source-applied+shift*full
                max_internal_norm = max(max_internal_norm,float(np.linalg.norm(residual[P])))
                max_external_norm = max(max_external_norm,float(np.linalg.norm(residual[Q])))
                if response_score == 'coupling':
                    contribution = np.max(abs(columns)*abs(solution)[None,:],axis=1) if len(P) else np.zeros(dimension)
                else:
                    contribution = abs(residual)/np.maximum(abs(diagonal-shift),denominator_floor)
                scores = np.maximum(scores,contribution)
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
    if len(indices) == dimension:
        response = candidates
    else:
        snapshot = SimpleNamespace(integrals=candidates.integrals,metadata=candidates.metadata,theta=candidates.theta,excitations=candidates.ground_excitations,trotter_steps=candidates.trotter_steps,sector_threads=candidates.simulator.sector_threads)
        response = USCCResponse(snapshot,[excitations[i] for i in indices],dressing=candidates.dressing)
    return ResponseSelection(response,indices,history,converged,'threshold_converged' if converged else 'cycle_limit',property_threshold,response_threshold,response_score)
