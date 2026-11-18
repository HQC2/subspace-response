"""Finite-candidate +V/+X selection for fixed-ground-state USCC response.

property_threshold compares |b_a|; response_threshold compares the external
individual-CSF residual L2 norm across properties and unique frequency branches.
Screening may use dressed, bare or first-order commutator operators; selected
response equations always use U†HU. Convergence refers to the finite pool
and screening criterion.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from uscc_response import USCCResponse
from calculation_progress import emit
from lr_screening import ExternalResiduals
from excitation_neighborhood import ExcitationNeighborhood
from csf_space import CompactResponseSpace


COUPLING_MODES = {'coupling': 'dressed', 'bare-coupling': 'bare', 'linear-coupling': 'linear'}


def property_neighborhood_indices(candidates):
    """Occupation neighbors of HF and ground generators; no spin templates."""
    space = candidates.space
    sim = candidates.simulator
    hf = int(sim.indices[np.flatnonzero(sim.initial)[0]])
    neighborhood = ExcitationNeighborhood(hf, sim.n_modes, SimpleNamespace(allowed=lambda block: True),
                                          max_distance=1)
    neighborhood._update_candidates(candidates.ground_excitations)
    if isinstance(space, CompactResponseSpace):
        catalog = space.catalog
        included = np.repeat([b in neighborhood._candidates for b in catalog.blocks],np.diff(catalog.starts))
        return np.flatnonzero(included[space.indices])
    return np.array([i for i,e in enumerate(space.excitations) if e.block in neighborhood._candidates],dtype=int)


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
    property_candidate_dimension: int


def select_response(candidates, properties, *, property_threshold=1e-4,
                    response_threshold=1e-4, response_score='coupling', seed_rank=1, shifts=(0j,),
                    denominator_floor=.05, solver_tolerance=1e-6, max_cycles=50, maxiter=500):
    """Select from a response object's independent rank-limited candidate pool.

    U and its ground-state parameters remain fixed. A zero threshold includes
    the entire supplied pool. ``seed_rank=0`` allows property-only seeding.
    ``shifts`` contains omega+i*gamma; both z and -z are solved automatically.
    The +X modes are coupling (U†HU), bare-coupling (H), and linear-coupling
    (H+[H,A], A=sum(theta_mu*(T_mu-T_mu†))). Hamiltonian approximations apply
    only to external screening; selected-space solves and final response use U†HU.
    Each external CSF is scored by the L2 norm of
    b_Q-Ktilde_QP*x_P across properties and unique +/- frequency branches.
    Full cancellations are included. Approximate modes score approximate
    residuals, not error bounds on the dressed equations.

    Property screening uses the single-electron occupation neighborhood
    of HF and ground-state generators. Outside it, b_Q is approximated by zero
    in +V/+X screening. Couplings within the neighborhood
    and all selected-space RHS/final properties remain exact. +X still visits
    the full candidate pool and can select CSFs outside the property neighborhood.
    """
    if any(not np.isfinite(x) or x < 0 for x in (property_threshold,response_threshold)):
        raise ValueError('Selection thresholds must be finite and nonnegative')
    if response_score not in COUPLING_MODES:
        raise ValueError('response_score must be '+', '.join(COUPLING_MODES))
    if type(seed_rank) is not int or seed_rank < 0 or type(max_cycles) is not int or max_cycles < 1:
        raise ValueError('Expected a nonnegative seed rank and positive cycle limit')
    if type(maxiter) is not int or maxiter < 1:
        raise ValueError('Expected a positive response iteration limit')
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
    property_indices = property_neighborhood_indices(candidates)
    emit(candidates.progress, 'properties_start',
         dimension=len(property_indices), candidate_dimension=len(excitations))
    sources = np.zeros((len(properties),len(excitations)))
    for source,op in zip(sources,properties):
        source[property_indices] = candidates.property_vector(op,indices=property_indices)
    selected = {i for i,e in enumerate(excitations) if e.rank <= seed_rank}
    selected.update(np.flatnonzero(np.any(abs(sources) >= property_threshold, axis=0)).tolist())
    emit(candidates.progress, 'lr_seed', dimension=len(selected), candidate_dimension=len(excitations),
         description=f'complete ranks through {seed_rank} plus dipole coupling threshold {property_threshold:g}')
    history = []
    approximation = COUPLING_MODES[response_score]
    converged = False
    previous, previous_indices = None, None
    for cycle in range(max_cycles+1):
        P = np.array(sorted(selected),dtype=int)
        emit(candidates.progress, 'lr_cycle', cycle=cycle, dimension=len(P))
        max_internal_norm = 0.
        response = candidates._restrict(P,matrix_free=True)
        response._retain_solution_images = True
        if previous is not None:
            response._seed_from_response(previous,np.searchsorted(P,previous_indices))
        screen = ExternalResiduals(candidates,P)
        # Retain all RHS images only when the batch and its concatenation fit
        # a quarter of the LR workspace. Larger frequency sweeps stream as before.
        image_count = len(branches)*len(properties)
        batch_screen = isinstance(candidates.space,CompactResponseSpace) and (
            2*16*image_count*(candidates.simulator.dimension+len(excitations)) <= candidates.memory_bytes//4)
        images, image_sources = [], []
        screen_started = False
        for shift in branches:
            dtype = complex if shift.imag else float
            for source,op in zip(sources,properties):
                rhs = response.property_vector(op).astype(dtype)
                source[P] = rhs.real
                solution,residual_norm = response._solve(rhs,omega=shift.real,gamma=shift.imag,
                    tolerance=solver_tolerance,denominator_floor=denominator_floor,maxiter=maxiter)
                max_internal_norm = max(max_internal_norm,residual_norm)
                if not screen_started:
                    emit(candidates.progress,'screen_start',method=response_score)
                    screen_started = True
                if response_threshold != 0:
                    image = response._solution_image(rhs,shift,solver_tolerance) if approximation == 'dressed' else None
                    if image is None:
                        image = response.sector_images(solution[:,None],approximation=approximation)[:,0]
                    if batch_screen:
                        images.append(image)
                        image_sources.append(source)
                    else:
                        screen.add(image,source)
        if images:
            screen.add(np.column_stack(images),np.column_stack(image_sources),threshold=response_threshold)
        if response_threshold == 0:
            grown = set(range(len(excitations)))
        else:
            grown = selected | set(screen.passed(response_threshold))
        emit(candidates.progress,'residual_screen',maximum=screen.maximum,images=len(screen.norms),
             norm_bounds=screen.norms_are_bounds)
        added = grown-selected
        emit(candidates.progress, 'lr_growth', added=len(added), dimension=len(P), will_grow=cycle < max_cycles)
        history.append(dict(cycle=cycle,dimension=len(P),ranks={str(r):sum(excitations[i].rank == r for i in P) for r in sorted({e.rank for e in excitations})},max_internal_residual_norm=max_internal_norm,added=[excitations[i].id for i in sorted(added)]))
        history[-1].update(criterion='external-csf-norm',screening_operator=approximation)
        if screen.norms_are_bounds:
            history[-1].update(external_residual_norm_bounds=screen.norms,max_external_csf_norm_bound=screen.maximum)
        else:
            history[-1].update(external_residual_norms=screen.norms,max_external_csf_norm=screen.maximum)
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
    return ResponseSelection(response,indices,history,converged,'threshold_converged' if converged else 'cycle_limit',property_threshold,response_threshold,response_score,
                             len(property_indices))
