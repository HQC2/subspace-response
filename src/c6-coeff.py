#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np

molecules = {}
molecules['H2'] = [['H','H'], 
qml.numpy.array([
[0., 0., 0.],
[0., 0., 0.74],
    ], requires_grad=False), 0]
molecules['LiH'] = [['Li','H'],
qml.numpy.array([
[0.0,  0.0         ,  0.],
[0., 0., 1.6717072740],
    ], requires_grad=False), 0]
molecules['LiH x2'] = [['Li','H'],
qml.numpy.array([
[0.0,  0.0         ,  0.],
[0., 0., 1.6717072740],
    ], requires_grad=False)*2, 0]
molecules['H2O'] = [['O','H','H'],
qml.numpy.array([
[0.0,  0.0         ,  0.1035174918],
[0.0,  0.7955612117, -0.4640237459],
[0.0, -0.7955612117, -0.4640237459],
    ], requires_grad=False), 0]
molecules['H2O x2'] = [['O','H','H'],
qml.numpy.array([
[0.0,  0.0         ,  0.1035174918],
[0.0,  0.7955612117, -0.4640237459],
[0.0, -0.7955612117, -0.4640237459],
    ], requires_grad=False)*2, 0]
molecules['OH-'] = [['O','H'],
qml.numpy.array([
[0., 0., 0.],
[0., 0., 1.0156985934],
    ], requires_grad=False), -1]
molecules['OH- x2'] = [['O','H'],
qml.numpy.array([
[0., 0., 0.],
[0., 0., 1.0156985934],
    ], requires_grad=False)*2, -1]
molecules['NH3'] = [['N','H', 'H', 'H'],
qml.numpy.array([
[          0.00000,        0.00000,       -0.06972],
[          0.00000,       -0.93223,        0.32291],
[          0.80734,        0.46611,        0.32291],
[         -0.80734,        0.46611,        0.32291],
    ], requires_grad=False), 0]
molecules['NH3'] = [['N','H', 'H', 'H'],
qml.numpy.array([
[          0.00000,        0.00000,       -0.06972],
[          0.00000,       -0.93223,        0.32291],
[          0.80734,        0.46611,        0.32291],
[         -0.80734,        0.46611,        0.32291],
    ], requires_grad=False)*2, 0]

basis = 'STO-3G'
for molecule, (symbols, geometry, charge) in molecules.items():
    print(f'Molecule: {molecule}')
    geometry *= 1.8897259886
    ucc = uccsd.uccsd(symbols, geometry, charge, basis)
    ucc.ground_state()
    hdiag = ucc.hess_diag_approximate()

    dipx, dipy, dipz = ucc.property_gradient('int1e_r')

    # Gauss-Legendre quadrature
    t, w = np.polynomial.legendre.leggauss(12) 
    gamma0 = 0.3
    # t in [-1,1] -> gamma in [0, inf]
    gammas = gamma0 * (1-t)/(1+t) 

    omega = 0.0
    history = None
    alpha_im = []
    for gamma in gammas:
        for dip in [dipx, dipy, dipz]:
            resp_plus, history = solvers.davidson_response(ucc.hvp, dip, hdiag, omega=omega, gamma=gamma, history=history, verbose=True)
            alpha_im.append(np.dot(resp_plus.real, dip))

    # alpha_im has alpha xx, yy, zz at each gamma
    alpha_im = np.array(alpha_im).reshape(-1,3)
    alpha_im_iso = np.average(np.array(alpha_im), axis=1)
    c6_coeff = 3.0/np.pi * sum((2*gamma0)/(1+t)**2 * w * alpha_im_iso**2)
    print(f'Molecule: {molecule}')
    print('C6:', c6_coeff)
    for gamma, wi, ai in zip(gammas, w, alpha_im_iso):
        print(gamma, wi, ai)
