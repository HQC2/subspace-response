#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np
import time

symbols = ['H', 'H', 'H', 'H']
geometry = qml.numpy.array([
[0.0,  0.0,  0.0],
[0.0,  0.0,  2.0],
[1.5,  0.0,  0.0],
[1.5,  0.0,  2.0],
]
                 , requires_grad=False)
basis = 'STO-3G'
charge = 0

ucc = uccsd.uccsd(symbols, geometry, charge, basis)
ucc.ground_state()

hdiag = ucc.hess_diag_approximate(triplet=True)
w,v, dim = solvers.davidson_liu(ucc.hvp_triplet, hdiag, len(hdiag), tol=1e-3)
