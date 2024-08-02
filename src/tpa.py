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

# davidson_response returns (respose, history)
intx, inty, intz = ucc.m.intor('int1e_r')
def tpa_term(ucc, A, B, roots):
    # solve for excited states
    hdiag = ucc.hess_diag_approximate()
    omega, X = solvers.davidson_liu(ucc.hvp, hdiag, roots, tol=1e-4)
    
    # degenerate TPA (omega_1 = omega_f/2), solve Na(omega_f - omega_1) and Nb(-omega_1) 
    V_A = ucc.property_gradient(A)
    V_B = ucc.property_gradient(B)
    zero = np.zeros(len(hdiag))
    for f in range(roots):
        omega_f = omega[f]
        omega_1 = omega_f/2
        Xf = X[:,f] # normalize according to full eigenvalue problem

        N_a = solvers.davidson_response(ucc.hvp, V_A, hdiag, verbose=False, omega=omega_f-omega_1)[0].real
        N_b = solvers.davidson_response(ucc.hvp, V_B, hdiag, verbose=False, omega=-omega_1)[0].real
        N_a_minus = solvers.davidson_response(ucc.hvp, -V_A, hdiag, verbose=False, omega=-(omega_f-omega_1))[0].real
        N_b_minus = solvers.davidson_response(ucc.hvp, -V_B, hdiag, verbose=False, omega=-(-omega_1))[0].real

        V2_NaBX = -ucc.V2_contraction(B, N_a, N_a_minus, Xf, zero) 
        V2_NbAX = 0.5*(ucc.V2_contraction(A, N_b_minus, N_b, Xf, zero) + ucc.V2_contraction(A, Xf, zero, N_b_minus, N_b))

        E3_NaNbX = ucc.E3_contraction(N_a, N_a_minus, N_b_minus, N_b, Xf, zero)
        E3_NaXNb = ucc.E3_contraction(N_a, N_a_minus, Xf, zero, N_b_minus, N_b)

        S3_NaNbX = ucc.S3_contraction(N_a, N_a_minus, N_b_minus, N_b, Xf, zero)
        S3_NaXNb = ucc.S3_contraction(N_a, N_a_minus, Xf, zero, N_b_minus, N_b)

        S = V2_NaBX + V2_NbAX + E3_NaNbX + E3_NaXNb - omega_1*S3_NaNbX + omega_f*S3_NaXNb
        print(V2_NaBX , V2_NbAX , E3_NaNbX , E3_NaXNb, - omega_1*S3_NaNbX, - omega_f*S3_NaXNb)
        print(f'Root {f+1} omega/2={omega_f/2} S={S}')

print(tpa_term(ucc, intz, intz, 3))
