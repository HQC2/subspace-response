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


def davidson_response(A, b, hdiag, history=None, omega=None, gamma=None,
                      tol=1e-6, verbose=False, *, maxiter=500,
                      denominator_floor=1e-10):
    """Solve ``(A + (omega + i*gamma) I) x = b`` by response Davidson.

    The plus sign preserves this module's historical frequency convention.
    ``tol`` is an absolute residual tolerance. ``A`` may be a matrix or a
    vector callback; complex vectors are split for real-only Hessian callbacks.
    ``hdiag`` is an approximate unshifted diagonal, or None for no preconditioner.

    History contains orthonormal V and its *unshifted* AV, so the same operator
    can reuse it across sources and shifts. The caller must discard history
    when the operator or coordinate space changes. The bounded subspace uses
    conjugate projections, double orthogonalization, and solution restarts.
    Nonconvergence and singular systems raise RuntimeError, never return a
    silently unconverged solution.
    """
    b = np.asarray(b)
    if b.ndim != 1 or not np.all(np.isfinite(b)):
        raise ValueError('Expected a finite response source vector')
    n = len(b)
    if hdiag is not None:
        hdiag = np.asarray(hdiag)
        if hdiag.shape != b.shape or not np.all(np.isfinite(hdiag)):
            raise ValueError('Expected a finite diagonal matching the source')
    if not np.isfinite(tol) or tol <= 0 or type(maxiter) is not int or maxiter < 1:
        raise ValueError('Expected positive tolerance and iteration limit')
    if not np.isfinite(denominator_floor) or denominator_floor <= 0:
        raise ValueError('Expected a positive denominator floor')
    shift = complex(0. if omega is None else omega, 0. if gamma is None else gamma)
    if not np.isfinite(shift):
        raise ValueError('Expected finite frequency shifts')
    shift = shift if shift.imag else shift.real
    matrix = None if callable(A) else np.asarray(A)
    if matrix is not None and (matrix.shape != (n, n) or not np.all(np.isfinite(matrix))):
        raise ValueError('Expected a finite square response matrix')
    dtype = np.result_type(b, float, shift, hdiag if hdiag is not None else float,
                           matrix if matrix is not None else float)
    b = b.astype(dtype, copy=False)
    if not np.any(b):
        return np.zeros_like(b), history
    limit = min(n, 64)

    def matvec(v):
        if matrix is not None:
            result = matrix @ v
        elif np.iscomplexobj(v) and np.any(v.imag):
            result = np.asarray(A(v.real)) + 1j*np.asarray(A(v.imag))
        else:
            result = np.asarray(A(v.real if np.iscomplexobj(v) else v))
        if result.shape != b.shape or not np.all(np.isfinite(result)):
            raise ValueError('Operator returned an invalid response vector')
        return result

    if hdiag is None:
        safe = np.ones(n)
    else:
        diagonal = hdiag+shift
        magnitude = abs(diagonal)
        phase = np.divide(diagonal, magnitude, out=np.ones_like(diagonal), where=magnitude != 0)
        safe = np.where(magnitude >= denominator_floor, diagonal, denominator_floor*phase)

    if history is None:
        old_v = old_av = np.empty((n, 0), dtype=dtype)
    else:
        old_v, old_av = np.asarray(history['V']), np.asarray(history['AV'])
        if old_v.ndim != 2 or old_v.shape[0] != n or old_av.shape != old_v.shape or not np.all(np.isfinite(old_v)) or not np.all(np.isfinite(old_av)):
            raise ValueError('Davidson history does not match the response space')
        if old_v.shape[1] > limit:
            raise ValueError('Davidson history exceeds the bounded subspace')
        dtype = np.result_type(dtype, old_v, old_av)
    V = np.empty((n, limit), dtype=dtype)
    AV = np.empty_like(V)
    H = np.empty((limit, limit), dtype=dtype)
    m = 0

    def orthogonalize(v):
        v = np.array(v, dtype=np.result_type(v, V.dtype), copy=True)
        norm = scipy.linalg.norm(v)
        if not norm:
            return None
        v /= norm
        for _ in range(2):
            v -= V[:, :m] @ (V[:, :m].conj().T @ v)
        norm = scipy.linalg.norm(v)
        return v/norm if norm > 1e-12 else None

    def append(v, av=None):
        nonlocal V, AV, H, m
        image = matvec(v) if av is None else av
        promoted = np.result_type(V, v, image)
        if promoted != V.dtype:
            V, AV, H = V.astype(promoted), AV.astype(promoted), H.astype(promoted)
        V[:, m], AV[:, m] = v, image
        H[:m+1, m] = V[:, :m+1].conj().T @ image
        H[m, :m] = v.conj() @ AV[:, :m]
        m += 1

    # Histories produced by this implementation are already orthonormal. QR
    # also repairs the old routine's unnormalized starting vector without new
    # operator products, transforming AV by the same change of coordinates.
    if old_v.shape[1]:
        gram = old_v.conj().T @ old_v
        if np.allclose(gram, np.eye(len(gram)), atol=1e-12, rtol=0):
            V[:, :old_v.shape[1]], AV[:, :old_v.shape[1]] = old_v, old_av
            m = old_v.shape[1]
            H[:m, :m] = V[:, :m].conj().T @ AV[:, :m]
        else:
            q, r = np.linalg.qr(old_v, mode='reduced')
            if np.min(abs(r.diagonal())) < 1e-12:
                raise ValueError('Davidson history contains dependent vectors')
            V[:, :len(r)], AV[:, :len(r)] = q, scipy.linalg.solve_triangular(r.T, old_av.T, lower=True).T
            m = len(r)
            H[:m, :m] = V[:, :m].conj().T @ AV[:, :m]
    if not m:
        append(orthogonalize(b/safe))
    for iteration in range(maxiter):
        reduced = H[:m, :m].copy()
        reduced.flat[::m+1] += shift
        projected_b = V[:, :m].conj().T @ b
        try:
            coefficients = np.linalg.solve(reduced, projected_b)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(reduced, projected_b, rcond=None)[0]
        x = V[:, :m] @ coefficients
        ax = AV[:, :m] @ coefficients
        residual = b-ax-shift*x
        residual_norm = scipy.linalg.norm(residual)
        if verbose:
            print(f'Iteration {iteration+1:3d} residual norm: {residual_norm:6e}')
        if residual_norm <= tol:
            return x, dict(V=V[:, :m].copy(), AV=AV[:, :m].copy())
        if iteration+1 == maxiter:
            break
        if m == limit and m < n:
            # Retain the current solution and its image without an A product.
            norm = scipy.linalg.norm(x)
            m = 0
            if norm:
                append(x/norm, ax/norm)
        trial = orthogonalize(residual/safe)
        if trial is None:
            trial = orthogonalize(residual)
        if trial is None:
            # A singular reduced projection of an indefinite operator can
            # leave its residual inside V. Its cached image supplies a Krylov
            # direction without another expensive application of A.
            for i in range(m):
                trial = orthogonalize(AV[:, i])
                if trial is not None:
                    break
        if trial is None or m == n:
            break
        append(trial)
    raise RuntimeError(f'Response Davidson did not converge after {iteration+1} iterations (residual={residual_norm:.3e})')
