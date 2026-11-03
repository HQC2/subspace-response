#!/usr/bin/env python
"""Build the optional pybind11/OpenMP rotation module into src/.

Requires a C++17 compiler with OpenMP and pybind11 headers. No dependency
download or compilation occurs on import of the simulator.
"""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
import tempfile


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--pybind11-path',type=Path,help='Directory containing an optional local pybind11 installation')
    cli.add_argument('--portable',action='store_true',help='Disable CPU-specific compiler instructions')
    cli.add_argument('--openmp-runtime',choices=['auto','compiler'],default='auto',help='Prefer PySCF\'s bundled GNU OpenMP runtime to avoid independent thread teams')
    cli.add_argument('--pyscf-batch', choices=['auto', 'off'], default='auto', help='Enable batched sector contractions when sharing PySCF OpenMP')
    args=cli.parse_args()
    if args.pybind11_path:
        sys.path.insert(0,str(args.pybind11_path.resolve()))
    try:
        import pybind11
    except ImportError:
        cli.error('Install pybind11 in the build environment, or use --pybind11-path')
    root=Path(__file__).resolve().parents[1]
    destination=root/'src'/('_uscc_native'+sysconfig.get_config_var('EXT_SUFFIX'))
    (root/'build').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='uscc-native-',dir=root/'build') as directory:
        temporary=Path(directory)/destination.name
        command=shlex.split(os.environ.get('CXX','c++'))+[
            '-O3','-std=c++17','-shared','-fPIC','-fvisibility=hidden','-fopenmp','-ffp-contract=off',
            '-DNDEBUG','-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
            str(root/'native'/'uscc_native.cpp'),'-o',str(temporary),
        ]
        if not args.portable:
            command.append('-march=native')
        runtime='compiler'
        if args.openmp_runtime=='auto':
            version=subprocess.check_output(shlex.split(os.environ.get('CXX','c++'))+['--version'],text=True)
            if 'clang' not in version.lower():
                try:
                    import pyscf
                    candidates=sorted((Path(pyscf.__file__).parent/'lib').glob('libgomp*.so*'))
                except ImportError:
                    candidates=[]
                if candidates:
                    library=candidates[0].resolve()
                    # This library resolves GOMP symbols before the driver's
                    # implicit -lgomp; --as-needed then omits a second runtime.
                    command.extend([str(library),'-Wl,-rpath,'+str(library.parent)])
                    runtime='pyscf'
                    contraction=library.parent/'libfci.so'
                    if args.pyscf_batch=='auto' and contraction.is_file():
                        command.extend([str(contraction),'-DUSCC_PYSCF_BATCH=1'])
        digest=sha256(b''.join(p.read_bytes() for p in
                              (root/'native'/'uscc_native.cpp', root/'native'/'uscc_scalar.hpp',
                               root/'native'/'uscc_connections.hpp'))).hexdigest()
        command.append('-DUSCC_NATIVE_SOURCE_SHA256="'+digest+'"')
        command.append('-DUSCC_NATIVE_OPENMP_RUNTIME="'+runtime+'"')
        print(shlex.join(command),flush=True)
        subprocess.run(command,check=True)
        os.replace(temporary,destination)
    print(destination)


if __name__=='__main__':
    main()
