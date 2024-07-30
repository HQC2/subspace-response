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

# davidson_response returns (respose, history)
intx, inty, intz = ucc.m.intor('int1e_r')
def beta_term(ucc, A, B, C, omega_B, omega_C):
    V_A = ucc.property_gradient(A)
    V_B = ucc.property_gradient(B)
    V_C = ucc.property_gradient(C)

    resp_a = solvers.davidson_response(ucc.hvp, V_A, hdiag, verbose=False, omega=(omega_B+omega_C))[0].real
    resp_b = solvers.davidson_response(ucc.hvp, V_B, hdiag, verbose=False, omega=omega_B)[0].real
    resp_c = solvers.davidson_response(ucc.hvp, V_C, hdiag, verbose=False, omega=omega_C)[0].real

    resp_a_minus = solvers.davidson_response(ucc.hvp, -V_A, hdiag, verbose=False, omega=-(omega_B+omega_C))[0].real
    resp_b_minus = solvers.davidson_response(ucc.hvp, -V_B, hdiag, verbose=False, omega=-omega_B)[0].real
    resp_c_minus = solvers.davidson_response(ucc.hvp, -V_C, hdiag, verbose=False, omega=-omega_C)[0].real

    V2_aBc = ucc.V2_contraction(B, resp_a_minus, resp_a, resp_c, resp_c_minus)
    V2_aCb = ucc.V2_contraction(C, resp_a_minus, resp_a, resp_b, resp_b_minus)
    V2_bAc = -0.5*(ucc.V2_contraction(A, resp_b, resp_b_minus, resp_c, resp_c_minus) + ucc.V2_contraction(A, resp_c, resp_c_minus, resp_b, resp_b_minus))

    E3_abc = ucc.E3_contraction(resp_a_minus, resp_a, resp_b, resp_b_minus, resp_c, resp_c_minus)
    E3_acb = ucc.E3_contraction(resp_a_minus, resp_a, resp_c, resp_c_minus, resp_b, resp_b_minus)

    S3_abc = ucc.S3_contraction(resp_a_minus, resp_a, resp_b, resp_b_minus, resp_c, resp_c_minus)
    S3_acb = ucc.S3_contraction(resp_a_minus, resp_a, resp_c, resp_c_minus, resp_b, resp_b_minus)
    print('A@B[2]@C', V2_aBc)
    print('A@C[2]@B', V2_aCb)
    print('B@A[2]@C', V2_bAc)
    print('A@E3_jlm@B@C', E3_abc)
    print('A@E3_jml@B@C', E3_acb)
    print('-omega_1*A@S3_jlm@B@C', -omega_B*S3_abc)
    print('-omega_2*A@S3_jml@B@C', -omega_C*S3_acb)
    return V2_aBc + V2_aCb + V2_bAc + E3_abc + E3_acb - omega_B*S3_abc - omega_C*S3_acb

print(beta_term(ucc, intz, intz, intz, 0.1, 0.3))
