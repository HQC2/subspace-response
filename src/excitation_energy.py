#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np
import time

symbols = ['H', 'H', 'H', 'H']
xyz = qml.numpy.array([
[0.0,  0.0,  0.0],
[0.0,  0.0,  2.0],
[1.5,  0.0,  0.0],
[1.5,  0.0,  2.0],
]
                 , requires_grad=False)
basis = 'STO-3G'
charge = 0
roots = 2

#thetas = []
# for _ in range(100):
#     xyz = qml.numpy.random.rand(4,3)*2.0
#     theta0 = uccsd.uccsd_ground_state(symbols, xyz, charge, basis=basis)
#     thetas.append(theta0[[7,9,11]])
# 
# np.savetxt('data', np.array(thetas))


hdiag = uccsd.hess_diag_approximate(symbols, xyz, charge, basis=basis)
theta0 = uccsd.uccsd_ground_state(symbols, xyz, charge, basis=basis)
hvp = uccsd.uccsd_hvp(symbols, xyz, charge, theta0, basis=basis)

op = np.zeros(len(theta0))
print('s-squared of ground state', uccsd.uccsd_spin_squared(symbols, xyz, charge, theta0, op, basis=basis))
#for i in range(len(theta0)):
#    op[i] = 1.0
#    print(f's2({i})', uccsd.uccsd_spin_squared(symbols, xyz, charge, theta0, op, basis=basis))
#    op[:] = 0.0
#
def fun(x):
    op = np.zeros(len(theta0))
    op[7:11] = x
    s2 = uccsd.uccsd_spin_squared(symbols, xyz, charge, theta0, op, basis=basis)
    op[:] = 0
    print(x, s2)
    return 1*(s2 + (np.linalg.norm(x)-1)**2)
#
#import scipy
#res = scipy.optimize.minimize(fun, method='slsqp', x0=[0,0,0,0])
#print(res.x)

t1 = time.time()
w,v = solvers.davidson_liu(hvp, hdiag, len(theta0), tol=1e-3)
for i in range(len(v)):
    op = v[:,i]
    for j, o in enumerate(op):
        if (np.abs(o)>0.1): 
            print(j, o, end=' ')
    print()
    print('s-squared', i, uccsd.uccsd_spin_squared(symbols, xyz, charge, theta0, op, basis=basis))
    print()
