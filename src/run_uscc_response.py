#!/usr/bin/env python
"""USCC ground state followed by state-transfer dipole response on its active orbitals."""
import json
from pathlib import Path
import sys

import numpy as np

from run_uscc import cli_progress, excitation_rank, load_ground_state, parser as ground_state_parser, run_ground_state
from response_selection import COUPLING_MODES, select_response
from uscc_response import USCCResponse


def parser():
    # Extend the same parser so new ground-state options are inherited here.
    cli = ground_state_parser(description=__doc__)
    cli.set_defaults(output_dir=Path('uscc-response-output'))
    response = cli.add_argument_group('dipole response (frequencies in hartree, properties in atomic units)')
    response.add_argument('--response-only', action='store_true', help='Use --restart without ground-state selection or optimization')
    response.add_argument('--response-max-rank', type=excitation_rank, default='unlimited', help='Independent response candidate rank, or unlimited (default: %(default)s)')
    response.add_argument('--components', nargs='+', choices=['x', 'y', 'z'], default=['x', 'y', 'z'], help='Dipole components defining both the tensor and selected space')
    response.add_argument('--frequencies', nargs='+', type=float, default=[0.], help='Real frequencies omega (default: static, 0)')
    response.add_argument('--damping', type=float, default=0., help='Nonnegative imaginary shift gamma (default: %(default)s)')
    response.add_argument('--response-tolerance', type=float, default=1e-6, help='Relative residual tolerance for selection and final linear solves (default: %(default)s)')
    response.add_argument('--response-maxiter', type=int, default=500, help='Linear solver iteration limit (default: %(default)s)')
    response.add_argument('--lr-memory-mb', type=float, default=1024., help='LR workspace budget in MiB, excluding integrals and simulator plans (default: %(default)s)')
    selection = cli.add_argument_group('response operator selection (+V/+X; state-transfer LR equations are fixed)')
    selection.add_argument('--response-selection', choices=[*COUPLING_MODES, 'full'], default='coupling', help='+X external equation-residual screening using U†HU (coupling), H (bare-coupling), or H+[H,A] (linear-coupling), or full candidate pool. (default: %(default)s)')
    selection.add_argument('--property-threshold', type=float, default=1e-4, help='+V threshold on absolute dipole coupling (default: %(default)s)')
    selection.add_argument('--response-threshold', type=float, default=1e-4, help='+X threshold on external CSF residual norm over properties and unique frequency branches, in dipole units (default: %(default)s)')
    selection.add_argument('--response-seed-rank', type=int, default=1, help='Ranks always included in the seed; 0 uses +V alone (default: %(default)s)')
    selection.add_argument('--response-max-cycles', type=int, default=50, help='Response selection growth limit (default: %(default)s)')
    selection.add_argument('--denominator-floor', type=float, default=.05, help='Positive floor for inner-solver orbital-gap preconditioning (default: %(default)s)')
    return cli


def validate_args(args, cli):
    """Reject unsupported combinations before running a molecular calculation."""
    if args.response_only and not args.restart:
        cli.error('--response-only requires --restart')
    if len(set(args.components)) != len(args.components):
        cli.error('--components must not contain duplicates')
    if not all(np.isfinite(w) for w in args.frequencies):
        cli.error('--frequencies must be finite')
    for name in ('damping', 'property_threshold', 'response_threshold'):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            cli.error(f'--{name.replace("_", "-")} must be finite and nonnegative')
    for name in ('response_tolerance', 'denominator_floor'):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            cli.error(f'--{name.replace("_", "-")} must be finite and positive')
    for name in ('response_maxiter', 'response_max_cycles'):
        if getattr(args, name) < 1:
            cli.error(f'--{name.replace("_", "-")} must be positive')
    if args.response_seed_rank < 0:
        cli.error('--response-seed-rank must be nonnegative')
    if not np.isfinite(args.lr_memory_mb) or args.lr_memory_mb <= 0:
        cli.error('--lr-memory-mb must be finite and positive')


def calculate_response(model, args):
    """Evaluate static or frequency-dependent dipole polarizability tensors."""
    physical_rank = min(model.integrals.electrons, model.integrals.qubits-model.integrals.electrons)
    max_rank = physical_rank if args.response_max_rank is None else min(args.response_max_rank, physical_rank)
    progress = cli_progress(args, 'Polarizability LR')
    if progress:
        progress.message(f'State-transfer LR on the fixed USCC reference; effective maximum rank={max_rank}; '
                         f'selection={args.response_selection}. LR uses the fixed-sector simulator.')
        progress.message(f'Components={" ".join(args.components)}; frequencies={args.frequencies} hartree; '
                         f'damping={args.damping:g} hartree. Both frequency branches contribute.')
        progress.message(f'Selection and final linear solves: relative residual tolerance={args.response_tolerance:.3e}, '
                         f'max iterations={args.response_maxiter}; orbital-gap preconditioning.')
        if args.response_selection != 'full':
            progress.message(f'Selection settings: seed rank={args.response_seed_rank}, '
                             f'dipole threshold (+V)={args.property_threshold:g}, '
                             f'residual threshold (+X)={args.response_threshold:g}, '
                             f'growth limit={args.response_max_cycles}.')
            progress.message(f'Selection denominator floor={args.denominator_floor:g} hartree.')
    # The constructor requires a positive rank even for a fully occupied space.
    response = USCCResponse(model, max_rank=max(1, max_rank),
                            sector_threads=model.sector_threads, progress=progress, compact=True, memory_mb=args.lr_memory_mb)
    if progress:
        progress.message(f'Matrix-free actions with occupation-block CSF transforms; LR workspace budget={args.lr_memory_mb:g} MiB '
                         '(integrals and simulator plans excluded).')
    candidate_dimension = len(response.space)
    if progress:
        progress.message(f'Computing dipole integrals for components {" ".join(args.components)} in the saved active orbitals.')
    operators = response.dipole_integrals()[['xyz'.index(c) for c in args.components]]
    selected = None
    if args.response_selection != 'full':
        selected = select_response(
            response, operators, property_threshold=args.property_threshold,
            response_threshold=args.response_threshold, response_score=args.response_selection,
            seed_rank=args.response_seed_rank,
            shifts=[complex(w, args.damping) for w in args.frequencies],
            denominator_floor=args.denominator_floor,
            solver_tolerance=args.response_tolerance, max_cycles=args.response_max_cycles,
            maxiter=args.response_maxiter)
        response = selected.response
        print(f'Response operator selection: {selected.stop_reason}; {len(response.space)}/{candidate_dimension} CSFs', flush=True)
    elif progress:
        progress.message(f'Using the full pool of {candidate_dimension} CSFs; no operator screening.')
    print(f'Response reference E={response.energy:.12f}; '
          f'ground residual={response.ground_residual_norm:.3e}; '
          f'{len(response.space)} CSFs', flush=True)
    tensors = []
    for omega in args.frequencies:
        if progress:
            progress.message(f'Computing the polarizability tensor at omega={omega:g}, gamma={args.damping:g} hartree.')
        options = dict(
            omega=omega, gamma=args.damping, tolerance=args.response_tolerance,
            maxiter=args.response_maxiter)
        rows = []
        for i, a in enumerate(operators):
            row = []
            for j, b in enumerate(operators):
                if progress:
                    progress.message(f'Computing alpha_{args.components[i]}{args.components[j]}.')
                row.append(response.linear_response(a, b, **options))
            rows.append(row)
        tensor = np.array(rows)
        tensors.append(dict(omega=omega, gamma=args.damping,
                            real=tensor.real.tolist(), imag=tensor.imag.tolist()))
        print(f'omega={omega:g} gamma={args.damping:g}: polarizability ({" ".join(args.components)})\n{tensor}', flush=True)
    result = dict(
        method='state-transfer', reference='uscc',
        requested_max_rank=args.response_max_rank, effective_max_rank=max_rank,
        components=args.components, units=dict(frequency='hartree', polarizability='atomic units'),
        reference_energy=response.energy, ground_residual_norm=response.ground_residual_norm,
        sector_dimension=response.simulator.dimension, candidate_dimension=candidate_dimension,
        response_dimension=len(response.space), response_ids=list(response.space.ids),
        solver=dict(method='response-davidson',tolerance=args.response_tolerance,
                    maxiter=args.response_maxiter,memory_mb=args.lr_memory_mb),
        polarizabilities=tensors,
        selection=dict(method=args.response_selection, converged=True, stop_reason='full_pool'))
    if selected is not None:
        result['selection'] = dict(
            method=args.response_selection, converged=selected.converged, stop_reason=selected.stop_reason,
            indices=list(selected.indices), history=selected.history,
            criterion='external-csf-norm',
            property_threshold=args.property_threshold, response_threshold=args.response_threshold,
            seed_rank=args.response_seed_rank, max_cycles=args.response_max_cycles,
            denominator_floor=args.denominator_floor)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    validate_args(args, cli)
    try:
        progress = cli_progress(args, 'Workflow')
        if progress:
            ground = 'keep the checkpoint ground state fixed' if args.response_only else 'prepare and optimize the USCC ground state'
            progress.message(f'Plan: {ground}, then compute dipole polarizabilities '
                             f'for {" ".join(args.components)} at {args.frequencies} hartree '
                             f'with {args.response_selection} LR operator selection.')
        model, thresholds = load_ground_state(args, cli)
        if args.response_only:
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
        result = calculate_response(model, args)
        result['ground_state'] = dict(
            checkpoint=str(checkpoint.resolve()), optimized=not args.response_only,
            energy=ground_result['energy'], stop_reason=ground_result['stop_reason'],
            spin_adaptation=model.spin_adaptation, trotter_order=model.trotter_order,
            trotter_steps=model.trotter_steps)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir/'response.json'
        destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(f'Response saved to {destination}', flush=True)
        return 0 if result['selection']['converged'] else 1
    except (ValueError, RuntimeError, OSError, np.linalg.LinAlgError) as error:
        print(f'USCC response: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
