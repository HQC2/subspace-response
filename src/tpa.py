#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np


symbols = ['H', 'H', 'H', 'H']
geometry = qml.numpy.array([
[0.0, 0.0, 0.0],
[0.0, 0.0, 2.0],
[2.0, 0.0, 2.0],
[1.0, 0.0, 0.0],
                 ], requires_grad=False)*1.8897259886
charge = 2
basis = 'STO-3G'

ucc = uccsd.uccsd(symbols, geometry, charge, basis)
ucc.ground_state()
hdiag = ucc.hess_diag_approximate()

# solve for omega and excitation vectors
roots = 1
omega, X = solvers.davidson_liu(ucc.hvp, hdiag, roots)

# davidson_response returns (respose, history)
intx, inty, intz = ucc.m.intor('int1e_r')
def S0f_transition_moment(ucc, A, B, omega_f, Xf):
    V_A = ucc.property_gradient(A)
    V_B = ucc.property_gradient(B)

    omega_1 = omega_f / 2
    zero = np.zeros_like(Xf)

    
    history = None
    N_a, history = solvers.davidson_response(ucc.hvp, V_A, hdiag, verbose=False, history=history, omega=omega_f-omega_1)
    N_b, history = solvers.davidson_response(ucc.hvp, V_B, hdiag, verbose=False, history=history, omega=-omega_1)
    N_a_minus, history = solvers.davidson_response(ucc.hvp, -V_A, hdiag, verbose=False, history=history, omega=-(omega_f-omega_1))
    N_b_minus, history = solvers.davidson_response(ucc.hvp, -V_B, hdiag, verbose=False, history=history, omega=-(-omega_1))

    V2_NaBX = -ucc.V2_contraction(B, N_a, N_a_minus, Xf, zero) 
    V2_NbAX = 0.5*(ucc.V2_contraction(A, N_b_minus, N_b, Xf, zero) + ucc.V2_contraction(A, Xf, zero, N_b_minus, N_b))

    E3_NaNbX = ucc.E3_contraction(N_a, N_a_minus, N_b_minus, N_b, Xf, zero)
    E3_NaXNb = ucc.E3_contraction(N_a, N_a_minus, Xf, zero, N_b_minus, N_b)

    S3_NaNbX = ucc.S3_contraction(N_a, N_a_minus, N_b_minus, N_b, Xf, zero)
    S3_NaXNb = ucc.S3_contraction(N_a, N_a_minus, Xf, zero, N_b_minus, N_b)

    S = V2_NaBX + V2_NbAX + E3_NaNbX + E3_NaXNb - omega_1*S3_NaNbX + omega_f*S3_NaXNb
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
