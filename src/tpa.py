#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np
import sys

active_electrons = None
active_orbitals = None

with open(sys.argv[1], 'r') as f:
    f.readline()
    charge, mult, active_electrons, active_orbitals = f.readline().split()
    #charge, mult, *_ = f.readline().split()
    charge = int(charge)
    active_electrons = int(active_electrons)
    active_orbitals = int(active_orbitals)
symbols, geometry = uccsd.read_xyz(sys.argv[1])
geometry /= 0.529177249 # expect bohr
basis = 'cc-pVDZ'


ucc = uccsd.uccsd(symbols, geometry, charge, basis, active_electrons=active_electrons, active_orbitals=active_orbitals)
ucc.ground_state()
hdiag = ucc.hess_diag_approximate()

# solve for omega and excitation vectors
roots = 2
omega, X = solvers.davidson_liu(ucc.hvp_new, hdiag, roots)

# davidson_response returns (respose, history)
intx, inty, intz = ucc.m.intor('int1e_r')
def S0f_transition_moment(ucc, A, B, omega_f, Xf, histcache={}):
    # zero by symmetry?
    if np.allclose(A, 0.) or np.allclose(B, 0.):
        return 0.
    V_A = ucc.property_gradient(A)
    V_B = ucc.property_gradient(B)

    omega_1 = omega_f / 2
    zero = np.zeros_like(Xf)

    
    history = histcache if histcache else None
    N_a, history = solvers.davidson_response(ucc.hvp_new, V_A, hdiag, verbose=True, history=history, omega=omega_f-omega_1)
    N_b, history = solvers.davidson_response(ucc.hvp_new, V_B, hdiag, verbose=True, history=history, omega=-omega_1)
    N_a_minus, history = solvers.davidson_response(ucc.hvp_new, -V_A, hdiag, verbose=True, history=history, omega=-(omega_f-omega_1))
    N_b_minus, history = solvers.davidson_response(ucc.hvp_new, -V_B, hdiag, verbose=True, history=history, omega=-(-omega_1))
    if history is not None:
        histcache.update(history)

    V2_NaBX = -ucc.V2_contraction(B, N_a, N_a_minus, Xf, zero) 
    V2_NbAX = 0.5*(ucc.V2_contraction(A, N_b_minus, N_b, Xf, zero) + ucc.V2_contraction(A, Xf, zero, N_b_minus, N_b))

    E3_NaNbX = ucc.E3_contraction(N_a, N_a_minus, N_b_minus, N_b, Xf, zero)
    E3_NaXNb = ucc.E3_contraction(N_a, N_a_minus, Xf, zero, N_b_minus, N_b)

    S = V2_NaBX + V2_NbAX + E3_NaNbX + E3_NaXNb
    return S

# loop over excited states
for root in range(roots):
    omega_f = omega[root]
    Xf = X[:,root]

    S = np.zeros((3,3))
    S[0,0] = S0f_transition_moment(ucc, intx, intx, omega_f, Xf)
    S[0,1] = S[1,0] = S0f_transition_moment(ucc, intx, inty, omega_f, Xf)
    S[0,2] = S[2,0] =  S0f_transition_moment(ucc, intx, intz, omega_f, Xf)
    S[1,1] = S0f_transition_moment(ucc, inty, inty, omega_f, Xf)
    S[1,2] = S[2,1] = S0f_transition_moment(ucc, inty, intz, omega_f, Xf)
    S[2,2] = S0f_transition_moment(ucc, intz, intz, omega_f, Xf)

    Df = np.einsum('ii,jj->', S, S)/30.0
    Dg = np.einsum('ij,ij->', S, S)/30.0

    D_lin = 2*Df + 4*Dg
    D_circ = -2*Df + 6*Dg

    print(f'Root {root+1} {omega_f=} {S[0,0]=} {S[1,1]=} {S[2,2]=} {S[0,1]=} {S[0,2]=} {S[1,2]=} {Df=} {Dg=} {D_lin=} {D_circ=}')
