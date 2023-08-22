#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np

symbols = ['O', 'H', 'H']
geometry = qml.numpy.array([
[0.0000000000,   -0.0000000000,    0.0664432016],
[0.0000000000,    0.7532904501,   -0.5271630249],
[0.0000000000,   -0.7532904501,   -0.5271630249]
                ], requires_grad=False) * 1.8897259886
basis ='STO-3G'
charge = 0

symbols = ['H', 'H']
geometry = qml.numpy.array([
[0.0, 0.0, 0.],
[0.0, 0.0, 2.0],
                 ], requires_grad=False)
basis = 'STO-3G'
charge = 0

ucc = uccsd.uccsd(symbols, geometry, charge, basis)
ucc.ground_state()

dipx, dipy, dipz = ucc.property_gradient('int1e_r')

resp_x = solvers.cg(ucc.hvp, dipx, verbose=True)
print(np.dot(resp_x, dipx))
resp_y = solvers.cg(ucc.hvp, dipy, verbose=True)
print(np.dot(resp_y, dipy))
resp_z = solvers.cg(ucc.hvp, dipz, verbose=True)
print(np.dot(resp_z, dipz))
