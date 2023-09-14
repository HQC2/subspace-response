#!/usr/bin/env python

import numpy as np
import scipy
import time
from typing import Callable

def davidson_liu(hvp, hdiag, roots, tol=1e-3):
    """
    Solves for eigenvalues and eigenvectors of a hessian.

    Args:
        hvp (callable): hessian-vector product, function that implements the matrix-vector product of the hessian with a trial vector
        hdiag (array): (approximate) diagonal hessian elements
        roots (int): number of roots to solve for
        tol (float): convergence tolerance on the residual norm
    """
    # select initial unit vectors based on diagonal hessian
    grad_dim = len(hdiag)
    V = np.zeros((grad_dim, roots))
    V[np.argsort(hdiag)[:roots], np.arange(roots)] = 1.0

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

def davidson_liu_response(hvp, property_gradient, omega, hdiag, roots, tol=1e-3):
    """
    Solves for eigenvalues and eigenvectors of a hessian.

    Args:
        hvp (callable): hessian-vector product, function that implements the matrix-vector product of the hessian with a trial vector
        hdiag (array): (approximate) diagonal hessian elements
        roots (int): number of roots to solve for
        tol (float): convergence tolerance on the residual norm
        property_gradient (array): Right hand side in the response equation
        omega (float): frequency of operator
    """
    # select initial unit vectors based on diagonal hessian
    grad_dim = len(hdiag)
    V = np.zeros((grad_dim, roots))
    V[np.argsort(hdiag)[:roots], np.arange(roots)] = 1.0

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

def cg(A,b, maxiter=1000, tol=1e-5, omega=None, gamma=None, verbose=False):
    if np.allclose(b, 0.0, atol=1e-20):
        return b
    if isinstance(A, np.ndarray):
        matvec = lambda v: A@v
    elif isinstance(A, Callable):
        matvec = A
    else:
        raise ValueError
    if omega is not None:
        I = omega*np.ones(len(b))
        if gamma is not None:
            I = I - 1j*gamma*np.ones(len(b))
    residual = np.copy(b)
    residual_norm = np.dot(residual,residual)
    x = np.zeros_like(b)
    direction = np.zeros_like(residual)
    direction[:] = residual[:]
    for i in range(maxiter):
        if np.allclose(direction.imag, 0.):
            product = matvec(direction)
        else:
            product_real = matvec(direction.real)
            product_imag = matvec(direction.imag)
            product = product_real - 1j*product_imag
        if omega is not None:
            product = product - I*direction
        gamma = residual_norm / np.dot(direction.conj(), product)
        x = x + gamma * direction
        residual_norm_old = residual_norm
        residual = residual - gamma * product
        residual_norm = np.dot(residual.conj(), residual)
        if verbose:
            print(i, np.sqrt(residual_norm))
        if residual_norm < tol**2:
            return x
        beta = residual_norm / residual_norm_old
        direction = residual + beta*direction
    raise ValueError('Not converged')
