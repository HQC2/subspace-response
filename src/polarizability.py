#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np


#symbols = ['Be', 'H', 'H']
#geometry = qml.numpy.array([
#[0.0,  0.0         ,  0.],
#[0., 0., 1.3517358726],
#[0., 0., -1.3517358726],
#                 ], requires_grad=False)*1.8897259886
symbols = ['Li', 'H']
geometry = qml.numpy.array([
[0.0,  0.0         ,  0.],
[0., 0., 1.6717072740],
                 ], requires_grad=False)*1.8897259886 *2
#symbols = ['O', 'H', 'H']
#geometry = qml.numpy.array([
#[0.0,  0.0         ,  0.1035174918],
#[0.0,  0.7955612117, -0.4640237459],
#[0.0, -0.7955612117, -0.4640237459],
#                 ], requires_grad=False)*1.8897259886
#geometry *= 2 # stretched
charge = 0

#symbols = ['C', 'O']
#geometry = qml.numpy.array([
#[0.0,  0.0,          0.0],
#[0.0,  0.0,          1.13848546872],
#                 ], requires_grad=False)*1.8897259886
#geometry *= 2 # stretched

#symbols = ['N', 'H', 'H', 'H']
#geometry = qml.numpy.array([
#[          0.00000,        0.00000,       -0.06972],
#[          0.00000,       -0.93223,        0.32291],
#[          0.80734,        0.46611,        0.32291],
#[         -0.80734,        0.46611,        0.32291],
#                 ], requires_grad=False)*1.8897259886
#geometry *= 2 # stretched


basis = 'STO-3G'

ucc = uccsd.uccsd(symbols, geometry, charge, basis)
ucc.ground_state()
hdiag = ucc.hess_diag_approximate()
print(len(hdiag))

dipx, dipy, dipz = ucc.property_gradient('int1e_r')
# davidson_response returns (respose, history)
resp_x, resp_y, resp_z = [solvers.davidson_response(ucc.hvp, dip, hdiag)[0] for dip in [dipx, dipy, dipz]]
print('alpha_xx', np.dot(dipx, resp_x).real)
print('alpha_yy', np.dot(dipy, resp_y).real)
print('alpha_zz', np.dot(dipz, resp_z).real)
