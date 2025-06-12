#!/usr/bin/env python

import sys
import uccsd
import solvers
import pennylane as qml
import numpy as np

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
basis = 'STO-3G'


ucc = uccsd.uccsd(symbols, geometry, charge, basis, active_electrons=active_electrons, active_orbitals=active_orbitals)
ucc.ground_state()
hdiag = ucc.hess_diag_approximate()

# davidson_response returns (respose, history)
intx, inty, intz = ucc.m.intor('int1e_r')
def beta_term(ucc, A, B, C, omega_B, omega_C):
    V_A = ucc.property_gradient(A)
    V_B = ucc.property_gradient(B)
    V_C = ucc.property_gradient(C)
    
    history = None
    resp_a, history = solvers.davidson_response(ucc.hvp, V_A, hdiag, verbose=False, history=history, omega=(omega_B+omega_C))
    resp_b, history = solvers.davidson_response(ucc.hvp, V_B, hdiag, verbose=False, history=history, omega=omega_B)
    resp_c, history = solvers.davidson_response(ucc.hvp, V_C, hdiag, verbose=False, history=history, omega=omega_C)

    resp_a_minus, history = solvers.davidson_response(ucc.hvp, -V_A, hdiag, verbose=False, history=history, omega=-(omega_B+omega_C))
    resp_b_minus, history = solvers.davidson_response(ucc.hvp, -V_B, hdiag, verbose=False, history=history, omega=-omega_B)
    resp_c_minus, history = solvers.davidson_response(ucc.hvp, -V_C, hdiag, verbose=False, history=history, omega=-omega_C)

    V2_aBc = ucc.V2_contraction(B, resp_a_minus, resp_a, resp_c, resp_c_minus)
    V2_aCb = ucc.V2_contraction(C, resp_a_minus, resp_a, resp_b, resp_b_minus)
    V2_bAc = -0.5*(ucc.V2_contraction(A, resp_b, resp_b_minus, resp_c, resp_c_minus) + ucc.V2_contraction(A, resp_c, resp_c_minus, resp_b, resp_b_minus))

    E3_abc = ucc.E3_contraction(resp_a_minus, resp_a, resp_b, resp_b_minus, resp_c, resp_c_minus)
    E3_acb = ucc.E3_contraction(resp_a_minus, resp_a, resp_c, resp_c_minus, resp_b, resp_b_minus)

    print('A@B[2]@C', V2_aBc)
    print('A@C[2]@B', V2_aCb)
    print('B@A[2]@C', V2_bAc)
    print('A@E3_jlm@B@C', E3_abc)
    print('A@E3_jml@B@C', E3_acb)
    return V2_aBc + V2_aCb + V2_bAc + E3_abc + E3_acb

print(beta_term(ucc, intz, intz, intz, 0.1, 0.3))
