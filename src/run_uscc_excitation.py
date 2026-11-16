#!/usr/bin/env python
"""USCC ground state followed by singlet state-transfer LR excitation energies."""
import json
from pathlib import Path
import sys

import numpy as np
from pyscf.data.nist import HARTREE2EV

from excitation_selection import select_excitations
from response_selection import COUPLING_MODES
from run_uscc import cli_progress, excitation_rank, load_ground_state, parser as ground_state_parser, run_ground_state
from uscc_response import USCCResponse


def parser():
    cli = ground_state_parser(description=__doc__)
    cli.set_defaults(output_dir=Path('uscc-excitation-output'))
    excitation = cli.add_argument_group('singlet excitation energies (hartree and eV)')
    excitation.add_argument('--excitation-only', action='store_true', help='Use --restart without ground-state selection or optimization')
    excitation.add_argument('--roots', type=int, default=5, help='Lowest roots, capped at the candidate dimension (default: %(default)s)')
    excitation.add_argument('--excitation-max-rank', type=excitation_rank, default='unlimited', help='Independent excitation candidate rank, or unlimited (default: %(default)s)')
    excitation.add_argument('--excitation-tolerance', type=float, default=1e-9, help='Absolute eigenvector residual tolerance in hartree (default: %(default)s)')
    excitation.add_argument('--excitation-maxiter', type=int, default=500, help='Iterative eigensolver iteration limit (default: %(default)s)')
    excitation.add_argument('--lr-memory-mb', type=float, default=1024., help='LR workspace budget in MiB, excluding integrals and simulator plans (default: %(default)s)')
    excitation.add_argument('--excitation-max-subspace', type=int, help='Davidson subspace cap; also bounded by --lr-memory-mb')
    selection = cli.add_argument_group('excitation operator selection (+X; state-transfer LR equations are fixed)')
    selection.add_argument('--excitation-selection', choices=['full', *COUPLING_MODES], default='full', help='Full pool, or external Ritz-residual screening using U†HU, H, or H+[H,A] (default: %(default)s)')
    selection.add_argument('--excitation-threshold', type=float, default=1e-4, help='+X threshold on external block residual norm over all roots, in hartree (default: %(default)s)')
    selection.add_argument('--excitation-seed', choices=['rank', 'cis'], default='rank', help='Seed for selected calculations: complete ranks or PySCF singlet CIS amplitudes (default: %(default)s)')
    selection.add_argument('--excitation-seed-rank', type=int, default=1, help='Ranks included with --excitation-seed rank (default: %(default)s)')
    selection.add_argument('--cis-roots', type=int, help='CIS roots contributing to the seed (default: --roots, capped at the singles dimension)')
    selection.add_argument('--cis-threshold', type=float, default=.05, help='Absolute normalized CIS amplitude cutoff; largest component of each root is always included (default: %(default)s)')
    selection.add_argument('--excitation-max-cycles', type=int, default=50, help='Selection growth limit (default: %(default)s)')
    return cli


def validate_args(args, cli):
    if args.excitation_only and not args.restart:
        cli.error('--excitation-only requires --restart')
    for name in ('roots', 'excitation_maxiter', 'excitation_max_cycles', 'excitation_seed_rank'):
        if getattr(args, name) < 1:
            cli.error(f'--{name.replace("_", "-")} must be positive')
    if args.excitation_max_subspace is not None and args.excitation_max_subspace < 1:
        cli.error('--excitation-max-subspace must be positive')
    if args.cis_roots is not None and args.cis_roots < 1:
        cli.error('--cis-roots must be positive')
    if not np.isfinite(args.excitation_tolerance) or args.excitation_tolerance <= 0:
        cli.error('--excitation-tolerance must be finite and positive')
    if not np.isfinite(args.excitation_threshold) or args.excitation_threshold < 0:
        cli.error('--excitation-threshold must be finite and nonnegative')
    if not np.isfinite(args.cis_threshold) or not 0 <= args.cis_threshold <= 1:
        cli.error('--cis-threshold must be between zero and one')
    if not np.isfinite(args.lr_memory_mb) or args.lr_memory_mb <= 0:
        cli.error('--lr-memory-mb must be finite and positive')


def calculate_excitations(model, args):
    physical_rank = min(model.integrals.electrons, model.integrals.qubits-model.integrals.electrons)
    max_rank = physical_rank if args.excitation_max_rank is None else min(args.excitation_max_rank, physical_rank)
    progress = cli_progress(args, 'Excitation LR')
    if progress:
        progress.message(f'Singlet state-transfer LR on the fixed USCC reference; requested roots={args.roots}, '
                         f'effective maximum rank={max_rank}; selection={args.excitation_selection}. '
                         'LR uses the fixed-sector simulator.')
        progress.message(f'Eigensolver: absolute residual tolerance={args.excitation_tolerance:.3e} hartree, '
                         f'iteration limit={args.excitation_maxiter}.')
        if args.excitation_selection != 'full':
            progress.message(f'Selection settings: block residual threshold={args.excitation_threshold:g} hartree, '
                             f'growth limit={args.excitation_max_cycles}; selected equations always use U†HU.')
            if args.excitation_seed == 'cis':
                progress.message(f'CIS seed: PySCF singlet TDA on saved active-space integrals, '
                                 f'roots={args.cis_roots or args.roots} (capped at the singles dimension), '
                                 f'normalized amplitude cutoff={args.cis_threshold:g}.')
            else:
                progress.message(f'Rank seed: include all CSFs through excitation rank {args.excitation_seed_rank}.')
    response = USCCResponse(model, max_rank=max(1, max_rank), sector_threads=model.sector_threads, progress=progress, compact=True, memory_mb=args.lr_memory_mb,
                            max_subspace=args.excitation_max_subspace)
    if progress:
        progress.message(f'Matrix-free actions with occupation-block CSF transforms; LR workspace budget={args.lr_memory_mb:g} MiB '
                         '(integrals and simulator plans excluded).')
    candidate_dimension = len(response.space)
    selection = dict(method='full', converged=True, stop_reason='full_pool')
    if progress and args.roots > candidate_dimension:
        progress.message(f'Root request capped to the {candidate_dimension} available candidate CSFs.')
    if args.excitation_selection == 'full':
        if progress:
            progress.message(f'Using the full pool of {candidate_dimension} CSFs; no operator screening.')
        energies, vectors = response.eigenpairs(args.roots, tolerance=args.excitation_tolerance,
                                               maxiter=args.excitation_maxiter)
    else:
        selected = select_excitations(
            response, roots=args.roots, excitation_threshold=args.excitation_threshold,
            excitation_score=args.excitation_selection, seed=args.excitation_seed,
            seed_rank=args.excitation_seed_rank, cis_roots=args.cis_roots,
            cis_threshold=args.cis_threshold, tolerance=args.excitation_tolerance,
            maxiter=args.excitation_maxiter, max_cycles=args.excitation_max_cycles)
        response, energies, vectors = selected.response, selected.energies, selected.vectors
        selection = dict(method=args.excitation_selection, converged=selected.converged,
                         stop_reason=selected.stop_reason, indices=list(selected.indices),
                         history=selected.history, seed=selected.seed,
                         criterion='external-block-frobenius',threshold=args.excitation_threshold, max_cycles=args.excitation_max_cycles)
        print(f'Excitation operator selection: {selected.stop_reason}; '
              f'{len(response.space)}/{candidate_dimension} CSFs', flush=True)
    print(f'Excitation reference E={response.energy:.12f}; '
          f'ground residual={response.ground_residual_norm:.3e}; '
          f'{len(response.space)} CSFs', flush=True)
    states = []
    for root, (energy, vector) in enumerate(zip(energies, vectors.T), 1):
        residual = float(response._eigen_result.residual_norms[root-1])
        states.append(dict(root=root, energy_hartree=float(energy), energy_ev=float(energy*HARTREE2EV),
                           residual_norm=residual, coefficients=vector.tolist()))
        print(f'Root {root}: {energy:.12f} hartree  {energy*HARTREE2EV:.8f} eV  '
              f'|r|={residual:.3e}', flush=True)
    return dict(method='state-transfer', reference='uscc', spin='singlet',
                requested_roots=args.roots, roots=len(energies),
                requested_max_rank=args.excitation_max_rank, effective_max_rank=max_rank,
                units=dict(energy='hartree', energy_ev='eV', residual='hartree'),
                reference_energy=response.energy, ground_residual_norm=response.ground_residual_norm,
                sector_dimension=response.simulator.dimension, candidate_dimension=candidate_dimension,
                excitation_dimension=len(response.space), excitation_ids=list(response.space.ids),
                solver=dict(method='block-davidson',tolerance=args.excitation_tolerance,
                    maxiter=args.excitation_maxiter,memory_mb=args.lr_memory_mb,
                    subspace_limit=response._eigen_result.subspace_limit,
                    iterations=response._eigen_result.iterations,products=response._eigen_result.products,
                    restarts=response._eigen_result.restarts),
                excitations=states, selection=selection)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    validate_args(args, cli)
    try:
        progress = cli_progress(args, 'Workflow')
        if progress:
            ground = 'keep the checkpoint ground state fixed' if args.excitation_only else 'prepare and optimize the USCC ground state'
            progress.message(f'Plan: {ground}, then compute {args.roots} lowest singlet LR excitation energies '
                             f'with {args.excitation_selection} operator selection.')
        model, thresholds = load_ground_state(args, cli)
        if args.excitation_only:
            progress = cli_progress(args, 'Ground state')
            if progress:
                progress.message('Keeping the checkpoint reference fixed; skipping ground-state selection and optimization.')
            ground_result = dict(energy=float(model.expectation()), stop_reason=model.stop_reason)
            checkpoint = args.restart
        else:
            ground_result = run_ground_state(model, thresholds, args)
            if ground_result['stop_reason'] == 'optimizer_failure':
                return 1
            checkpoint = args.output_dir/'checkpoint.json'
        result = calculate_excitations(model, args)
        result['ground_state'] = dict(
            checkpoint=str(checkpoint.resolve()), optimized=not args.excitation_only,
            energy=ground_result['energy'], stop_reason=ground_result['stop_reason'],
            spin_adaptation=model.spin_adaptation, trotter_order=model.trotter_order,
            trotter_steps=model.trotter_steps)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir/'excitations.json'
        destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(f'Excitation energies saved to {destination}', flush=True)
        return 0 if result['selection']['converged'] else 1
    except (ValueError, RuntimeError, OSError, np.linalg.LinAlgError) as error:
        print(f'USCC excitation energies: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
