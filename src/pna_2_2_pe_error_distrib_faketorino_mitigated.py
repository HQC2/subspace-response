#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np
import time
from functools import partial
import sys
import os
from mitigation import rem_mitigate, get_confusion_matrix


symbols, geometry = uccsd.read_xyz('calcs/pna_in_water_sphere_80.xyz')
geometry *= 1.8897261245650618
basis = '6-31+G*'
charge = 0
ucc = uccsd.uccsd(symbols, geometry, charge, basis, active_electrons=2, active_orbitals=2, PE='calcs/pna_in_water_sphere_80.pot')
ucc.ground_state()

hdiag = ucc.hess_diag_approximate(triplet=False)

ucc.circuit.diff_method = 'parameter-shift'
ucc.circuit_exc.diff_method = 'parameter-shift'

wref = 4.64191
hvp = ucc.hvp
scheme = '7-point'
h = 0.5
ucc.hvp = partial(hvp, h=h, scheme=scheme)


full_shotcount = int(1e5)
fd_shotcount = int(full_shotcount / 6) # 7-point takes 6x shots
repeats = 1

# setup fake torino
from qiskit_aer.noise import NoiseModel
import qiskit_ibm_runtime
faketorino = qiskit_ibm_runtime.fake_provider.FakeTorino()
noise_model = NoiseModel.from_backend(faketorino)

noisy_device = qml.device("qiskit.aer", wires=ucc.qubits, shots=full_shotcount, noise_model=noise_model, optimization_level=0)
noisy_device_for_fd = qml.device("qiskit.aer", wires=ucc.qubits, shots=fd_shotcount, noise_model=noise_model, optimization_level=0)

# get M0 confusion matrix
if os.path.isfile('confusion.npy'):
    confusion = np.load('confusion.npy')
else:
    print('Building dimsedutmatrix')
    confusion = get_confusion_matrix(ucc, noisy_device)
    np.save('confusion.npy', confusion)

ucc.circuit_operator_stateprep.device = noisy_device
ucc.circuit_exc_operator.device = noisy_device
ucc.circuit.device = noisy_device
ucc.circuit_operator.device = noisy_device

ucc.circuit_operator_stateprep.diff_method = "parameter-shift"
ucc.circuit_exc_operator.diff_method = "parameter-shift"
ucc.circuit.diff_method = "parameter-shift"
ucc.circuit_operator.diff_method = "parameter-shift"

ucc.circuit_operator_stateprep = rem_mitigate(ucc.circuit_operator_stateprep, confusion)
ucc.circuit_exc_operator= rem_mitigate(ucc.circuit_exc_operator, confusion)
ucc.circuit = rem_mitigate(ucc.circuit, confusion)
ucc.circuit_operator = rem_mitigate(ucc.circuit_operator, confusion)

print('Superpostion state hvp approach:')
for repeat in range(repeats):
    w, v = solvers.davidson_liu(ucc.hvp_new, hdiag, 2, tol=1e300)
    omega = w[0]*27.211399
    print(f'hvp-super: shots={full_shotcount} scheme={scheme} h={h} err={omega-wref}', flush=True)
