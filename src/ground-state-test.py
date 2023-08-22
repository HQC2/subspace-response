#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np
import sys


symbols = ['H', 'H']
geometry = qml.numpy.array([
[0.0,  0.0,  0.0],
[0.0,  0.0,  2.0],
                ], requires_grad=False)

basis = 'STO-3G'
charge = 0

ucc = uccsd.uccsd(symbols, geometry, charge, basis)
ucc.ground_state()
