#!/usr/bin/env python

import numpy as np

def davidson_liu(hvp, hdiag, roots, grad_dim):
    # select initial unit vectors based on diagonal hessian
    V = np.zeros((grad_dim, roots))
    V[np.argsort(hdiag)[:roots], np.arange(roots)] = 1.0

    tol = 1e-3
    AV = hvp(V)
    for i in range(100):
        S = V.T @ AV
        L, Z = np.linalg.eigh(S)
        L = L[:roots]
        Z = Z[:,:roots]
        X = V@Z
        AX = AV@Z
        r = (AX - L[None,:]*X)
        print(f'Iter    Root    Residual        Eigenvalue (au, eV)')
        for k in range(len(L)):
            print(f'{i+1}       {k+1}       {np.linalg.norm(r[:,k]):.6e}    {L[k]:.6f} {L[k]*27.2114:.6f}')
        if (all(np.linalg.norm(r, axis=0) < tol)):
            return L, X
        delta = np.array([1/(L[k] - hdiag) * r[:,k] for k in range(len(L))]).T
        delta /= np.linalg.norm(delta, axis=0)

        new_vs = []
        for k in range(len(L)):
            q = delta[:,k] - V@(V.T@delta[:,k])
            qnorm = np.linalg.norm(q)
            if qnorm > 1e-3:
                vp = q / qnorm
                V = np.hstack([V, vp[:,None]])
                new_vs.append(vp)
        new_vs = np.array(new_vs).T
        AV = np.hstack([AV, hvp(new_vs)])
    raise ValueError('Convergence not reached')

