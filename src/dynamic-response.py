#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np


symbols = ['N', 'H', 'H', 'H']
geometry = qml.numpy.array([
[          0.00000,        0.00000,       -0.06972],
[          0.00000,       -0.93223,        0.32291],
[          0.80734,        0.46611,        0.32291],
[         -0.80734,        0.46611,        0.32291],
                 ], requires_grad=False)*1.8897259886
basis = 'STO-3G'
charge = 0

#symbols = ['O', 'H', 'H']
#
#geometry = qml.numpy.array([
#[0.0,  0.0         ,  0.1035174918],
#[0.0,  0.7955612117, -0.4640237459],
#[0.0, -0.7955612117, -0.4640237459],
#                 ], requires_grad=False)*1.8897259886


ucc = uccsd.uccsd(symbols, geometry, charge, basis)
ucc.ground_state()
hdiag = ucc.hess_diag_approximate()

dipx, dipy, dipz = ucc.property_gradient('int1e_r')

gamma = 0.004556
gamma = 0.0
history = None
for dip in [dipx, dipy, dipz]:
    history = None
    for omega in np.arange(14.8, 15.4, 0.002):
        resp_plus, history = solvers.davidson_response(ucc.hvp, dip, hdiag, omega=omega, gamma=gamma, history=history, verbose=True)
        resp_minus, history = solvers.davidson_response(ucc.hvp, -dip, hdiag, omega=-omega, gamma=gamma, history=history, verbose=True)
        # hermitian operator
        resp_imag = resp_plus.imag + resp_minus.imag
        resp_real = resp_plus.real - resp_minus.real
        print(omega, 0.5*np.dot(resp_real, dip), 0.5*np.dot(resp_imag, dip), f'dim_V={history["V"].shape}', flush=True)
