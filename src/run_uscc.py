#!/usr/bin/env python
"""Command-line ground-state USCC, with self-contained JSON checkpoints."""
import argparse
import json
from pathlib import Path
import sys
from uscc import USCC


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
    cli.add_argument('xyz', nargs='?', help='XYZ geometry; angstrom by default')
    cli.add_argument('--restart', type=Path, help='Resume a self-contained checkpoint')
    cli.add_argument('--unit', choices=['bohr','angstrom'])
    cli.add_argument('--basis')
    cli.add_argument('--charge', type=int)
    cli.add_argument('--active-electrons', type=int)
    cli.add_argument('--active-orbitals', type=int)
    cli.add_argument('--max-rank', type=int)
    cli.add_argument('--seed', choices=['screened', 'sd'])
    cli.add_argument('--trotter-steps', type=int)
    cli.add_argument('--exhaustive', action='store_true', default=None, help='Audit the complete small-system pool; zero threshold includes it all')
    cli.add_argument('--orbital-irreps', nargs='+', type=int, help='Optional Abelian XOR irrep IDs in active orbital order')
    cli.add_argument('--growth', choices=['paper','recursive'], help='Paper algorithm through quadruples (default), or higher-rank extension')
    cli.add_argument('--optimizer', choices=['slsqp','bfgs'])
    cli.add_argument('--thresholds', nargs='+', type=float)
    cli.add_argument('--gradient-tolerance', type=float)
    cli.add_argument('--energy-tolerance', type=float)
    cli.add_argument('--maxiter', type=int)
    cli.add_argument('--max-cycles', type=int, default=50)
    cli.add_argument('--max-operators', type=int)
    cli.add_argument('--backend', choices=['sector','pennylane'], help='Ideal fixed-sector simulator (default) or circuit reference')
    cli.add_argument('--sector-threads', type=int, help='OpenMP threads for sector PySCF contractions; default uses PySCF settings')
    cli.add_argument('--device', help='PennyLane simulator, default lightning.qubit')
    cli.add_argument('--output-dir', type=Path, default=Path('uscc-output'))
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.restart:
            forbidden = ('xyz','unit','basis','charge','active_electrons','active_orbitals','max_rank','seed','trotter_steps','exhaustive','orbital_irreps','growth')
            if any(getattr(args,key) is not None for key in forbidden):
                cli.error('Restart uses the saved molecule, orbitals, ansatz, and selection configuration')
            model = USCC.from_checkpoint(args.restart, device=args.device, backend=args.backend, sector_threads=args.sector_threads)
            thresholds = args.thresholds if args.thresholds is not None else model.schedule
            if thresholds is None:
                thresholds = (.01,.001,.0001)
        else:
            if args.xyz is None:
                cli.error('Provide an XYZ geometry or --restart')
            symbols, geometry = read_xyz(args.xyz)
            model = USCC.from_molecule(symbols, geometry, charge=0 if args.charge is None else args.charge, basis=args.basis or 'sto-3g', unit=args.unit or 'angstrom', active_electrons=args.active_electrons, active_orbitals=args.active_orbitals, max_rank=4 if args.max_rank is None else args.max_rank, seed=args.seed or 'sd', trotter_steps=1 if args.trotter_steps is None else args.trotter_steps, device=args.device or 'lightning.qubit', orbital_irreps=args.orbital_irreps, exhaustive=bool(args.exhaustive), backend=args.backend or 'sector', growth=args.growth or 'paper', sector_threads=args.sector_threads)
            thresholds = args.thresholds if args.thresholds is not None else (.01,.001,.0001)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        def progress(event):
            if event['event'] == 'optimization':
                print(f"E={event['energy']:.12f} |g|={event['gradient_norm']:.3e} S2={event['spin_squared']:.3e} ranks={event['ranks']} success={event['success']}", flush=True)
            elif event['event'] == 'growth':
                print(f"epsilon={event['epsilon']:g}: added {len(event['added'])} operators, ranks={event['ranks']}", flush=True)
        result = model.run(thresholds, checkpoint=args.output_dir/'checkpoint.json', max_cycles=args.max_cycles, max_operators=args.max_operators, gradient_tolerance=args.gradient_tolerance, energy_tolerance=args.energy_tolerance, maxiter=args.maxiter, optimizer=args.optimizer, callback=progress)
        (args.output_dir/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(f"Stopped: {result['stop_reason']}; energy={result['energy']:.12f}")
        return 1 if result['stop_reason'] == 'optimizer_failure' else 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f'USCC: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
