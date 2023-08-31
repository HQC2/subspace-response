#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np


symbols = ['H', 'H']#, 'H', 'H']
geometry = qml.numpy.array([
[0.0000000000,   -0.0000000000,    0.0],
[0.0000000000,   -0.0000000000,    2.0],
#[1.5000000000,   -0.0000000000,    0.0],
#[1.5000000000,   -0.0000000000,    2.0],
                 ], requires_grad=False)
basis = 'STO-3G'
charge = 0

ucc = uccsd.uccsd(symbols, geometry, charge, basis)
#theta = np.array([-1.35066821e-16, -4.54632328e-17, -4.86482370e-17, -1.42047871e-16,
#  1.03048404e-01,  2.26491572e-16,  1.13066447e-16, -1.87589773e-01,
# -5.11526101e-02,  4.97979006e-02,  3.49274247e-16,  2.14070553e-01,
#  1.15892459e-16,  9.26284955e-02,])
#ucc.theta = theta
ucc.ground_state()

dipx, dipy, dipz = ucc.property_gradient('int1e_r')

#resp_x = solvers.cg(ucc.hvp, dipx, verbose=True)
#print(np.dot(resp_x, dipx))
#resp_y = solvers.cg(ucc.hvp, dipy, verbose=True)
#print(np.dot(resp_y, dipy))
for omega in np.arange(0, 1.4, 0.01):
    resp_z = solvers.cg(ucc.hvp, dipz, omega=omega, gamma=0.1, verbose=False)
    print(omega, np.dot(resp_z, dipz).real, np.dot(resp_z, dipz).imag)
