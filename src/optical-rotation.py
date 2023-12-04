#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np
import sys

def get_response_fct(prop1, prop2, num_imag, omega, history,diag=True):
  """
  Obtain response function by parsing two integrals
  """

  if diag == False:
     raise NameError("Only diagonal response function is implemented")

  diag_resp = []
  for p1, p2 in zip(prop1, prop2):
      # Calculate linear response vector of prop1
      resp_plus, history = solvers.davidson_response(ucc.hvp, p1, hdiag, omega=omega, history=history, verbose=True)
      resp_minus, history = solvers.davidson_response(ucc.hvp, -p1, hdiag, omega=-omega, history=history, verbose=True)
      resp = resp_plus -(-1)**num_imag * resp_minus 
      resp_fct = -(-1)**num_imag * 0.5*np.dot(resp,p2)
      diag_resp.append(resp_fct.real)
      print(omega, resp_fct, f'dim_V={history["V"].shape}', flush=True)

  return diag_resp

# H4 geometry from Kumar paper
symbols = ['H', 'H', 'H', 'H']
geometry = qml.numpy.array([
  [0.000000000000 ,   -0.750000000000,    -0.324759526419],
  [-0.375000000000,    -0.760000000000,     0.324759526419],
  [0.000000000000,     0.750000000000,    -0.324759526419],
  [0.375000000000,     0.850000000000,     0.45],
                 ], requires_grad=False)*1.8897259886 # \AA to a.u.
basis = 'STO-3G'
charge = 2 # to make it FCI for Dalton comparison

ucc = uccsd.uccsd(symbols, geometry, charge, basis)
ucc.ground_state()
hdiag = ucc.hess_diag_approximate()

#Get property gradients
dips_x= ucc.property_gradient('int1e_r') # V_\mu
dips_p= ucc.property_gradient('int1e_ipovlp', approach='statevector') # V_p
mags= ucc.property_gradient('int1e_cg_irxp', approach='statevector')  # V_m

omega = 0.07619285815529173 # 589nm in a.u.
history = None

# Length Gauge: <<\mu,m>>
mu_m = get_response_fct(dips_x,mags,num_imag=1,omega=omega,history=history)

# Velocity Gauge: <<p,m>>
p_m = get_response_fct(dips_p,mags,num_imag=2,omega=omega,history=history)

# Modified velocity gauge, term 2: 
p_m_0 = get_response_fct(dips_p,mags,num_imag=2,omega=0,history=history)

print("Length gauge")
print("xx:", mu_m[0])
print("yy:", mu_m[1])
print("zz:", mu_m[2])

print("Velocity gauge:")
print("xx:", p_m[0])
print("yy:", p_m[1])
print("zz:", p_m[2])

print("Velocity gauge, term 2:")
print("xx:", p_m_0[0])
print("yy:", p_m_0[1])
print("zz:", p_m_0[2])
