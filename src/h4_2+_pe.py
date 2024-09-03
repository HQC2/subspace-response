#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np
import time

symbols, geometry = uccsd.read_xyz('calcs/H4.xyz')
geometry *= 1.8897261245650618

basis = 'STO-3G'
charge = 2
#roots = 5

ucc = uccsd.uccsd(symbols, geometry, charge, basis, active_electrons=2, active_orbitals=4, PE='calcs/nh3_step_2220.pot')
hdiag = ucc.hess_diag_approximate(triplet=False)
ucc.ground_state(min_method='slsqp')

result = solvers.davidson_liu(ucc.hvp_new, hdiag, 5, tol=1e-6)
transition_densities = ucc.transition_density(result[1])
transition_densities = ucc.mf.mo_coeff @ transition_densities @ ucc.mf.mo_coeff.T
print(np.einsum('kmn,xmn->kx', transition_densities, ucc.m.intor('int1e_r')))
