#!/usr/bin/env python

import uccsd
import pennylane as qml
import numpy as np

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

omega = 0.07619285815529173 # 589nm in a.u.
history = None

G_len, G_veloc, G_modveloc = ucc.get_OR(omega=omega)
print("G' length: ", G_len)
print("G' velocity: ", G_veloc)
print("G' modified velocity: ", G_modveloc)
