"""Eigenvector-driven operator selection for singlet state-transfer LR-USCC.

The +X score is the external residual L2 norm of each CSF over all requested unit-normalized
Ritz vectors. Only external screening may use bare or linearized couplings;
all selected-space eigenproblems use the correlated USCC Hamiltonian.
"""
from dataclasses import dataclass

import numpy as np
from pyscf import ao2mo, gto, lib, scf
from pyscf.tdscf.rhf import gen_tda_operation
from solvers import davidson_eigenpairs
from lr_screening import ExternalResiduals

from response_selection import COUPLING_MODES
from uscc_response import USCCResponse
from calculation_progress import emit


@dataclass
class ExcitationSelection:
    response: USCCResponse
    indices: tuple
    energies: np.ndarray
    vectors: np.ndarray
    history: list
    converged: bool
    stop_reason: str
    seed: dict


def cis_guess(candidates, roots=5, *, tolerance=1e-9, maxiter=500):
    """PySCF singlet RHF-TDA/CIS in the saved active space.

    Returns energies and unit-normalized coefficients in occupied/virtual
    spatial-orbital order, *not* PySCF's alpha-only (norm 1/sqrt(2)) amplitudes.
    Frozen-core effects are already in h_eff. No SCF is rerun and no molecular
    metadata is needed. Semicanonicalization within occupied/virtual blocks
    also supports noncanonical saved orbitals; amplitudes are rotated back.
    """
    integrals = candidates.integrals
    norb, nocc = integrals.qubits//2, integrals.electrons//2
    dimension = nocc*(norb-nocc)
    if type(roots) is not int or roots < 1 or not dimension:
        raise ValueError('CIS needs occupied and virtual orbitals and a positive root count')
    if not np.isfinite(tolerance) or tolerance <= 0 or type(maxiter) is not int or maxiter < 1:
        raise ValueError('CIS tolerance and iteration limit must be positive')
    roots = min(roots, dimension)
    progress = getattr(candidates, 'progress', None)
    emit(progress, 'eigen_start', kind='CIS', roots=roots, dimension=dimension,
         solver='block Davidson with PySCF TDA',
         tolerance=tolerance, maxiter=maxiter)
    # An in-core RHF object in an orthonormal active-orbital basis. PySCF's
    # response kernel uses these integrals for J/K without AO reconstruction.
    molecule = gto.M(verbose=0)
    molecule.nelectron = integrals.electrons
    mean_field = scf.RHF(molecule)
    mean_field.chkfile = None
    mean_field.get_hcore = lambda *args: integrals.one_body
    mean_field.get_ovlp = lambda *args: np.eye(norb)
    mean_field._eri = ao2mo.restore(8, candidates.simulator.chemist_eri, norb)
    occupations = np.zeros(norb)
    occupations[:nocc] = 2.
    density = np.diag(occupations)
    with lib.with_omp_threads(candidates.simulator.sector_threads):
        fock = integrals.one_body + mean_field.get_veff(dm=density)
        occupied_energies, occupied = np.linalg.eigh(fock[:nocc, :nocc])
        virtual_energies, virtual = np.linalg.eigh(fock[nocc:, nocc:])
        coefficients = np.zeros((norb, norb))
        coefficients[:nocc, :nocc] = occupied
        coefficients[nocc:, nocc:] = virtual
        mean_field.mo_coeff = coefficients
        mean_field.mo_occ = occupations
        mean_field.mo_energy = np.r_[occupied_energies, virtual_energies]
        mean_field.converged = True
        tda_action, diagonal = gen_tda_operation(mean_field, singlet=True)
        result = davidson_eigenpairs(lambda v: np.asarray(tda_action(v.T)).T, diagonal, roots,
            tolerance=tolerance,maxiter=maxiter,memory_bytes=getattr(candidates,'memory_bytes',1024*1024**2)//4,
            callback=None if progress is None else lambda e: emit(progress,'eigen_iteration',kind='CIS',**e))
        energies, vectors = result.energies, result.vectors
    for root,(energy,residual) in enumerate(zip(energies,result.residual_norms),1):
        emit(progress,'eigen_root',kind='CIS',root=root,energy=float(energy),residual_norm=float(residual))
    amplitudes = np.array([occupied @ v.reshape(nocc, norb-nocc) @ virtual.T
                           for v in vectors.T])
    return energies, amplitudes


def select_excitations(candidates, *, roots=5, excitation_threshold=1e-4,
                       excitation_score='coupling', seed='rank', seed_rank=1,
                       cis_roots=None, cis_threshold=.05, tolerance=1e-9,
                       maxiter=500, max_cycles=50):
    """Grow a common space for the lowest requested singlet LR roots.

    Rank seeds include every CSF through ``seed_rank``. CIS seeds include
    singles whose normalized absolute amplitude in any requested CIS root is
    at least ``cis_threshold``, plus each root's largest component. Lowest-gap
    CSFs supplement either seed if needed to represent all requested roots.
    A zero +X threshold
    recovers the full candidate pool. Convergence refers only to that pool and
    the chosen residual criterion; it does not certify missing disconnected roots.
    """
    if excitation_score not in COUPLING_MODES:
        raise ValueError('excitation_score must be '+', '.join(COUPLING_MODES))
    if seed not in ('rank', 'cis'):
        raise ValueError('Excitation seed must be rank or cis')
    if type(roots) is not int or roots < 1 or not len(candidates.space):
        raise ValueError('Expected a nonempty candidate pool and positive root count')
    if type(seed_rank) is not int or seed_rank < 1:
        raise ValueError('Seed rank must be positive')
    if cis_roots is not None and (type(cis_roots) is not int or cis_roots < 1):
        raise ValueError('CIS root count must be positive')
    if not np.isfinite(cis_threshold) or not 0 <= cis_threshold <= 1:
        raise ValueError('CIS amplitude threshold must be between zero and one')
    if not np.isfinite(excitation_threshold) or excitation_threshold < 0:
        raise ValueError('Excitation threshold must be finite and nonnegative')
    if not np.isfinite(tolerance) or tolerance <= 0 or type(maxiter) is not int or maxiter < 1:
        raise ValueError('Eigensolver tolerance and iteration limit must be positive')
    if type(max_cycles) is not int or max_cycles < 1:
        raise ValueError('Selection cycle limit must be positive')
    roots = min(roots, len(candidates.space))
    excitations = candidates.space.excitations
    amplitudes = None
    seed_info = dict(method=seed)
    if seed == 'rank':
        selected = {i for i, e in enumerate(excitations) if e.rank <= seed_rank}
        seed_info['rank'] = seed_rank
    else:
        cis_energies, amplitudes = cis_guess(candidates, roots if cis_roots is None else cis_roots,
                                           tolerance=tolerance, maxiter=maxiter)
        # Only magnitudes select operators, so the canonical CSF phase relative
        # to a spin-summed spatial single has no effect on the seed.
        nocc = candidates.integrals.electrons//2
        scores = np.max(abs(amplitudes), axis=0)
        keep = scores >= cis_threshold
        for amplitude in amplitudes:
            keep.flat[np.argmax(abs(amplitude))] = True
        selected = {i for i, e in enumerate(excitations) if e.rank == 1
                    and keep[e.spatial_holes[0], e.spatial_particles[0]-nocc]}
        seed_info.update(threshold=cis_threshold, requested_roots=cis_roots,
                         energies_hartree=cis_energies.tolist(), roots=len(cis_energies))
    initial = set(selected)
    for i in np.argsort(candidates.orbital_diagonal(), kind='stable'):
        if len(selected) >= roots:
            break
        selected.add(int(i))
    seed_info.update(indices=sorted(selected), dimension=len(selected),
                     supplemented_indices=sorted(selected-initial))
    emit(candidates.progress, 'lr_seed', dimension=len(selected), candidate_dimension=len(excitations),
         description=f'{seed} seed; {len(selected-initial)} CSFs supplemented to represent the requested roots')
    history = []
    approximation = COUPLING_MODES[excitation_score]
    converged = False
    previous, previous_indices = None, None
    for cycle in range(1,max_cycles+1):
        indices = np.asarray(sorted(selected),dtype=int)
        emit(candidates.progress,'lr_cycle',cycle=cycle-1,dimension=len(indices))
        response = candidates._restrict(indices,matrix_free=True)
        guesses = None
        if previous is not None:
            guesses = response._seed_from_eigenpairs(previous,np.searchsorted(indices,previous_indices))
        elif amplitudes is not None:
            # Canonical CSF phase relative to (a†_alpha i_alpha +
            # a†_beta i_beta)|HF>/sqrt(2), including its fermionic phase.
            guesses = np.zeros((len(indices),len(amplitudes)))
            for j,i in enumerate(indices):
                e = excitations[i]
                if e.rank == 1:
                    alpha = next(t for t in e.terms if t.holes[0]%2 == 0)
                    guesses[j] = np.sqrt(2)*alpha.coefficient*amplitudes[:,e.spatial_holes[0],e.spatial_particles[0]-nocc]
        energies,vectors = response.eigenpairs(roots,tolerance=tolerance,maxiter=maxiter,guesses=guesses)
        emit(candidates.progress,'screen_start',method=excitation_score)
        screen = ExternalResiduals(candidates,indices)
        if excitation_threshold == 0:
            grown = set(range(len(excitations)))
        else:
            # Reuse fresh dressed images from the mandatory residual audit.
            # Each uncached or approximate image is one root action, never a
            # selected basis-column construction.
            for root,vector in enumerate(vectors.T):
                image = response._eigen_images.get(root) if approximation == 'dressed' else None
                if image is None:
                    image = response.sector_images(vector[:,None],approximation=approximation)[:,0]
                screen.add(image)
            grown = selected | set(screen.passed(excitation_threshold))
        added = grown-selected
        emit(candidates.progress,'residual_screen',maximum=screen.maximum,images=len(screen.norms))
        emit(candidates.progress,'lr_growth',added=len(added),dimension=len(indices),will_grow=cycle<max_cycles)
        result = response._eigen_result
        history.append(dict(cycle=cycle,dimension=len(indices),energies_hartree=energies.tolist(),
            residual_norms=result.residual_norms.tolist(),criterion='external-csf-norm',
            screening_operator=approximation,external_residual_norms=screen.norms,
            ranks={str(r):sum(excitations[i].rank==r for i in indices)
                   for r in sorted({e.rank for e in excitations})},
            added=[excitations[i].id for i in sorted(added)],
            eigen_iterations=result.iterations,eigen_products=result.products,
            eigen_restarts=result.restarts,eigen_subspace_limit=result.subspace_limit))
        history[-1]['max_external_csf_norm'] = screen.maximum
        if not added:
            converged = True
            break
        if cycle == max_cycles:
            break
        previous,previous_indices = response,indices
        selected = grown
    return ExcitationSelection(response,tuple(sorted(selected)),energies,vectors,history,
        converged,'threshold_converged' if converged else 'cycle_limit',seed_info)
