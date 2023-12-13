#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np


symbols = ['H', 'H', 'H', 'H']
geometry = qml.numpy.array([
  [   -0.05092677388561  ,   -0.72869825426711,      0.00000000000007],
  [   0.05092677388561 ,     0.72869825426711 ,     0.00000000000007],
  [   0.89262010034093    , -0.92385699172443 ,    -0.00000000000007],
  [   -0.89262010034093    ,  0.92385699172444,     -1.00000000000007],
                 ], requires_grad=False)*1.8897259886
basis = 'STO-3G'
charge = 2



ucc = uccsd.uccsd(symbols, geometry, charge, basis)
ucc.ground_state()
hdiag = ucc.hess_diag_approximate()

dipx, dipy, dipz = ucc.property_gradient('int1e_r')
magx, magy, magz = ucc.property_gradient('int1e_cg_irxp', approach='statevector')


omega = 0.07619285815529173
history = None
for dip, mag in zip([dipx, dipy, dipz], [magx, magy, magz]):
    resp_plus, history = solvers.davidson_response(ucc.hvp, dip, hdiag, omega=omega, history=history, verbose=True)
    resp_minus, history = solvers.davidson_response(ucc.hvp, -dip, hdiag, omega=-omega, history=history, verbose=True)
    # hermitian operator
    resp = resp_plus + resp_minus
    print(omega, -0.5*np.dot(resp, mag), f'dim_V={history["V"].shape}', flush=True)
