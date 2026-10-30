#!/usr/bin/env python
"""Ground-state USCC or SPQE selection with VQE/PQE and JSON checkpoints."""
import argparse
import json
from pathlib import Path
import sys
from uscc import USCC


class StoreExplicit(argparse.Action):
    """Record supplied options so restart can distinguish them from defaults."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace._provided = getattr(namespace, '_provided', set()) | {self.dest}


def read_xyz(filename):
    lines = Path(filename).read_text().splitlines()
    try:
        count = int(lines[0])
        if count < 1 or len(lines) < count+2:
            raise ValueError
        atoms = [line.split() for line in lines[2:count+2]]
        if any(len(atom) != 4 for atom in atoms):
            raise ValueError
        return [atom[0] for atom in atoms], [[float(x) for x in atom[1:]] for atom in atoms]
    except (IndexError, ValueError) as error:
        raise ValueError('Expected an XYZ file with atom count, comment, and element/x/y/z rows') from error


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.register('action', None, StoreExplicit)
    cli.add_argument('xyz', nargs='?', help='XYZ geometry; angstrom by default')
    cli.add_argument('--restart', type=Path, help='Resume a self-contained checkpoint')
    cli.add_argument('--unit', choices=['angstrom', 'bohr'], default='angstrom')
    cli.add_argument('--basis', default='sto-3g')
    cli.add_argument('--charge', type=int, default=0)
    cli.add_argument('--active-electrons', type=int)
    cli.add_argument('--active-orbitals', type=int)
    cli.add_argument('--max-rank', type=int, default=4, help='Maximum excitation rank for recursive operator growth (default: %(default)s)')
    cli.add_argument('--seed', choices=['screened', 'sd'], default='screened', help='Initial operator seed (default: %(default)s)')
    cli.add_argument('--selection', choices=['uscc', 'spqe'], default='uscc', help='USCC amplitude/integral screening or SPQE residual selection in the single/double neighborhood')
    cli.add_argument('--trotter-steps', type=int, default=1)
    cli.add_argument('--orbital-irreps', nargs='+', type=int, help='Optional Abelian XOR irrep IDs in active orbital order')
    cli.add_argument('--optimizer', choices=['lbfgs', 'slsqp','bfgs','diis','pqe'], help='Default: VQE L-BFGS; pqe solves projected residual equations using gap updates and DIIS')
    cli.add_argument('--thresholds', nargs='+', type=float, help='Decreasing selection thresholds; SPQE uses an excluded residual L2 budget in hartree')
    cli.add_argument('--gradient-tolerance', type=float)
    cli.add_argument('--residual-tolerance', type=float, help='PQE active residual L2 tolerance (default: 1e-6 hartree)')
    cli.add_argument('--energy-tolerance', type=float)
    cli.add_argument('--maxiter', type=int)
    cli.add_argument('--max-cycles', type=int, default=50)
    cli.add_argument('--max-operators', type=int)
    cli.add_argument('--backend', choices=['sector','pennylane'], default='sector', help='Ideal fixed-sector simulator (default) or circuit reference')
    cli.add_argument('--sector-threads', type=int, help='OpenMP threads for sector PySCF contractions; default uses PySCF settings')
    cli.add_argument('--sector-kernels', choices=['auto','native','numba','numpy'], default='auto', help='Rotation kernels: auto prefers the optional C++ module, then Numba, then NumPy')
    cli.add_argument('--rotation-threads', type=int, help='Native rotation threads; default uses one for small sectors, otherwise up to four sector threads')
    cli.add_argument('--device', default='lightning.qubit', help='PennyLane simulator, default %(default)s')
    cli.add_argument('--output-dir', type=Path, default=Path('uscc-output'))
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    provided = vars(args).pop('_provided', set())
    print(args)
    try:
        if args.restart:
            forbidden = {'unit','basis','charge','active_electrons','active_orbitals','max_rank','seed','selection','trotter_steps','orbital_irreps'}
            if args.xyz is not None or provided & forbidden:
                cli.error('Restart uses the saved molecule, orbitals, ansatz, and selection configuration')
            overrides = {key: getattr(args, key) for key in ('device','backend','sector_threads','sector_kernels','rotation_threads') if key in provided}
            model = USCC.from_checkpoint(args.restart, **overrides)
            thresholds = args.thresholds if args.thresholds is not None else model.schedule
            if thresholds is None:
                thresholds = (.01,.001,.0001)
        else:
            if args.xyz is None:
                cli.error('Provide an XYZ geometry or --restart')
            symbols, geometry = read_xyz(args.xyz)
            model = USCC.from_molecule(
                symbols, geometry, charge=args.charge, basis=args.basis,
                unit=args.unit, active_electrons=args.active_electrons,
                active_orbitals=args.active_orbitals, max_rank=args.max_rank,
                seed=args.seed, selection=args.selection, trotter_steps=args.trotter_steps,
                device=args.device, orbital_irreps=args.orbital_irreps,
                backend=args.backend, sector_threads=args.sector_threads,
                sector_kernels=args.sector_kernels,
                rotation_threads=args.rotation_threads)
            thresholds = args.thresholds if args.thresholds is not None else (.01,.001,.0001)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        def progress(event):
            if event['event'] == 'optimization':
                total = sum(event['ranks'].values())
                convergence = (f"|r|={event['residual_norm']:.3e}" if event['optimizer'] == 'pqe'
                               else f"|g|={event['gradient_norm']:.3e}")
                print(f"E={event['energy']:.12f} {convergence} S2={event['spin_squared']:.3e} ranks={event['ranks']} {total=} success={event['success']}", flush=True)
            elif event['event'] == 'growth':
                total = sum(event['ranks'].values())
                print(f"epsilon={event['epsilon']:g}: added {len(event['added'])} operators, ranks={event['ranks']} {total=}", flush=True)
        result = model.run(thresholds, checkpoint=args.output_dir/'checkpoint.json', max_cycles=args.max_cycles, max_operators=args.max_operators, gradient_tolerance=args.gradient_tolerance, residual_tolerance=args.residual_tolerance, energy_tolerance=args.energy_tolerance, maxiter=args.maxiter, optimizer=args.optimizer, callback=progress)
        (args.output_dir/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(f"Stopped: {result['stop_reason']}; energy={result['energy']:.12f}")
        return 1 if result['stop_reason'] == 'optimizer_failure' else 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f'USCC: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
