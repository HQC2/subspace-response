#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np


symbols = ['Be', 'H', 'H']
geometry = qml.numpy.array([
[0.0,  0.0         ,  0.],
[0., 0., 1.3517358726],
[0., 0., -1.3517358726],
                 ], requires_grad=False)*1.8897259886
basis = 'STO-3G'
charge = 0

ucc = uccsd.uccsd(symbols, geometry, charge, basis)
print(ucc.qubits)
ucc.ground_state()
hdiag = ucc.hess_diag_approximate()

dipx, dipy, dipz = ucc.property_gradient('int1e_r')
resp_x, resp_y, resp_z = [solvers.davidson_response(ucc.hvp, dip, hdiag) for dip in [dipx, dipy, dipz]]
print('alpha_xx', np.dot(dipx, resp_x).real)
print('alpha_yy', np.dot(dipy, resp_y).real)
print('alpha_zz', np.dot(dipz, resp_z).real)
