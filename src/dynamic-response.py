#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np


symbols = ['H', 'H']
geometry = qml.numpy.array([
[0.0000000000,   -0.0000000000,    0.0],
[0.0000000000,   -0.0000000000,    2.0],
                 ], requires_grad=False)
basis = 'STO-3G'
charge = 0

ucc = uccsd.uccsd(symbols, geometry, charge, basis)
ucc.ground_state()
hdiag = ucc.hess_diag_approximate()

dipx, dipy, dipz = ucc.property_gradient('int1e_r')

dipz = dipz
for omega in np.arange(0., 1.2, 0.01):
    resp_z_plus = solvers.davidson_response(ucc.hvp, dipz, hdiag, omega=omega, gamma=0.02)
    resp_z_minus = solvers.davidson_response(ucc.hvp, -dipz, hdiag, omega=-omega, gamma=0.02)
    # hermitian operator
    resp_z_imag = resp_z_plus.imag + resp_z_minus.imag
    resp_z_real = resp_z_plus.real - resp_z_minus.real
    print(omega, 0.5*np.dot(resp_z_real, dipz), 0.5*np.dot(resp_z_imag, dipz))
