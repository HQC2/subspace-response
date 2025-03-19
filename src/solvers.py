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
        Z = Z[:, :roots]
        X = V @ Z
        AX = AV @ Z
        r = (AX - L[None, :]*X)
        print(f'{r.shape}')
        print(f'Iter    Root    Residual        Eigenvalue (au, eV)')
        for k in range(len(L)):
            print(f'{i+1}       {k+1}       {np.linalg.norm(r[:,k]):.6e}    {L[k]:.6f} {L[k]*27.2114:.6f}')
        if (all(np.linalg.norm(r, axis=0) < tol)):
            print(f'At solution, AV dimension: {AV.shape}')
            return L, X
        delta = np.array([1/(L[k] - hdiag)*r[:, k] for k in range(len(L))]).T
        delta_norms = np.linalg.norm(delta, axis=0)
        delta /= delta_norms

        new_vs = []
        for k in range(len(L)):
            q = delta[:, k] - V @ (V.T @ delta[:, k])
            qnorm = np.linalg.norm(q)
            residual_norm = np.linalg.norm(r[:,k])
            print(f'State={k+1} {residual_norm=} {qnorm=}')
            if qnorm > 1e-3 and residual_norm > tol:
                vp = q/qnorm
                V = np.hstack([V, vp[:, None]])
                new_vs.append(vp)
        new_vs = np.array(new_vs).T
        AV = np.hstack([AV, hvp(new_vs)])
    raise ValueError('Convergence not reached')


def cg(A, b, maxiter=100, tol=1e-4, omega=None, gamma=None, guess=None, verbose=False):
    if np.allclose(b, 0.0, atol=1e-20):
        return b
    if omega is not None:
        I = omega*np.ones(len(b))
        if gamma is not None:
            I = I + 1j*gamma*np.ones(len(b))
    else:
        I = np.zeros(len(b))
    if isinstance(A, np.ndarray):
        matvec = lambda v: A @ v - I*v
    elif isinstance(A, Callable):
        matvec = lambda v: A(v) - I*v
    else:
        raise ValueError

    if guess is None:
        guess = np.zeros_like(b)
    x = guess
    residual = np.copy(b) - matvec(x)
    residual_norm = np.dot(residual, residual)
    direction = np.zeros_like(residual)
    direction[:] = residual[:]
    for i in range(maxiter):
        if np.allclose(direction.imag, 0.):
            product = matvec(direction)
        else:
            product_real = matvec(direction.real)
            product_imag = matvec(direction.imag)
            product = product_real + 1j*product_imag
        gamma = residual_norm/np.dot(direction.conj(), product)
        x = x + gamma*direction
        residual_norm_old = residual_norm
        residual = residual - gamma*product
        residual_norm = np.dot(residual.conj(), residual)
        if verbose:
            print(i, np.sqrt(residual_norm))
        if residual_norm < tol**2:
            if verbose:
                print(f'Converged in {i+1} iterations (residual norm = {residual_norm**0.5})')
            return x
        beta = residual_norm/residual_norm_old
        direction = residual + beta*direction
    print('Not converged...')
    return x


def davidson_response(A, b, hdiag, history=None, omega=None, gamma=None, tol=1e-6, verbose=False):
    if np.allclose(b, 0.0, atol=1e-20):
        return b, history

    def matvec(v, A=A, omega=omega, gamma=gamma):
        if np.allclose(v.imag, 0.):
            Av = A(v)
        else:
            Av_real = A(v.real)
            Av_imag = A(v.imag)
            Av = Av_real + 1j*Av_imag
        return Av

    I = np.ones(len(hdiag))
    S = np.zeros_like(I)
    if omega is not None:
        S = S - omega*I
    if gamma is not None:
        S = S - 1j*gamma*I

    diagonal_guess = b/(hdiag)
    diagonal_guess /= np.linalg.norm(diagonal_guess)
    if history is not None:
        V = history['V']
        AV = history['AV']
        SV = np.zeros_like(V, dtype=np.complex128)
        for i in range(V.shape[1]):
            SV[:, i] = S*V[:, i]
    else:
        V = diagonal_guess.astype(np.complex128).reshape(-1, 1)
        AV = np.zeros_like(V, dtype=np.complex128)
        SV = np.zeros_like(V, dtype=np.complex128)
        AV[:, 0] = matvec(V[:, 0])
        SV[:, 0] = S*V[:, 0]

    bred = V.T @ b
    for i in range(100):
        Ered = V.T @ (AV - SV)
        xred = np.linalg.solve(Ered, bred)
        x = xred @ V.T
        residual = b - (xred @ AV.T - S*x)
        #       (equivalent to)
        #       residual = b - (matvec(x) - S*x)
        if verbose:
            print(f'Iteration {i+1:3d} residual norm: {np.linalg.norm(residual):6e}')
        if np.linalg.norm(residual) < tol:
            history = {'V': V, 'AV': AV}
            return x, history
        delta = residual/(hdiag)
        delta = delta/np.linalg.norm(delta)
        vnew = delta - V @ (V.T @ delta)
        vnew = vnew/np.linalg.norm(vnew)
        V = np.hstack([V, vnew[:, None]])
        bred = V.T @ b
        AV = np.hstack([AV, matvec(vnew)[:, None]])
        SV = np.hstack([SV, (S*vnew)[:, None]])
    raise ValueError('Not converged')
