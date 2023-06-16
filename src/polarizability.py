#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np

symbols = ['O', 'H', 'H']
xyz = qml.numpy.array([
[0.0000000000,   -0.0000000000,    0.0664432016],
[0.0000000000,    0.7532904501,   -0.5271630249],
[0.0000000000,   -0.7532904501,   -0.5271630249]
                ], requires_grad=False) * 1.8897259886


hdiag = uccsd.hess_diag_approximate(symbols, xyz, 0)
theta0 = uccsd.uccsd_ground_state(symbols, xyz, 0)
hvp = uccsd.uccsd_hvp(symbols, xyz, 0, theta0)

dipx, dipy, dipz = uccsd.uccsd_dipole_property_gradient(symbols, xyz, 0, theta0)

if dipx is not None:
    resp_x = solvers.cg(hvp, dipx)
    print(np.dot(resp_x, dipx))
if dipy is not None:
    resp_y = solvers.cg(hvp, dipy)
    print(np.dot(resp_y, dipy))
if dipz is not None:
    resp_z = solvers.cg(hvp, dipz)
    print(np.dot(resp_z, dipz))

