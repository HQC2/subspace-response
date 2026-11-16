#!/usr/bin/env python

import numpy as np
import scipy
from typing import Callable
from dataclasses import dataclass


@dataclass
class DavidsonEigenResult:
    energies: np.ndarray
    vectors: np.ndarray
    residual_norms: np.ndarray
    iterations: int
    products: int
    restarts: int
    subspace_limit: int


def davidson_eigenpairs(matmat, diagonal, roots, *, tolerance=1e-9, maxiter=500,
                        max_subspace=None, memory_bytes=128*1024**2, guesses=None,
                        history=None, callback=None, audit=None, denominator_floor=1e-8):
    """Bounded block Davidson--Liu for the lowest Hermitian eigenpairs.

    Operators accept column batches, never a constructed Hamiltonian. Only V,
    KV and the small Ritz matrix are retained. History contains *valid* images
    in the current coordinates. The optional audit returns fresh operator
    images; callers can retain underlying sector images during that audit.
    """
    diagonal = np.asarray(diagonal, dtype=float)
    n = len(diagonal)
    if diagonal.ndim != 1 or not n or not np.all(np.isfinite(diagonal)):
        raise ValueError('Expected a nonempty finite Davidson diagonal')
    if type(roots) is not int or roots < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Expected positive root count and Davidson tolerance')
    if type(maxiter) is not int or maxiter < 1 or not np.isfinite(denominator_floor) or denominator_floor <= 0:
        raise ValueError('Expected a positive Davidson iteration limit and denominator floor')
    if not np.isfinite(memory_bytes) or memory_bytes <= 0:
        raise ValueError('Davidson memory budget must be positive')
    roots = min(roots, n)
    if max_subspace is not None and (type(max_subspace) is not int or max_subspace < min(n, roots+1)):
        raise ValueError('Davidson subspace must leave room for correction vectors')
    limit = min(n, max_subspace if max_subspace is not None else max(32, 6*roots+12))
    # Conservative complex-sized V/AV, Ritz and root/correction work accounting.
    while limit >= roots and 16*(2*n*limit+4*limit*limit+5*n*roots) > memory_bytes:
        limit -= 1
    if limit < min(n, roots+1):
        raise ValueError('Davidson memory budget cannot hold the requested roots and a correction subspace')
    rng = np.random.default_rng(29)
    initial, known = [], []
    if history is not None:
        hv, ha = np.asarray(history['V']), np.asarray(history['AV'])
        if hv.ndim != 2 or hv.shape[0] != n or ha.shape != hv.shape or not np.all(np.isfinite(hv)) or not np.all(np.isfinite(ha)):
            raise ValueError('Davidson eigenvector history does not match this space')
        initial.extend(hv.T)
        known.extend(ha.T)
    if guesses is not None:
        guesses = np.asarray(guesses)
        if guesses.ndim == 1:
            guesses = guesses[:, None]
        if guesses.ndim != 2 or guesses.shape[0] != n or not np.all(np.isfinite(guesses)):
            raise ValueError('Davidson guesses do not match this space')
        initial.extend(guesses.T)
        known.extend([None]*guesses.shape[1])
    dtype = np.result_type(float, *[v.dtype for v in initial])
    V = np.empty((n, limit), dtype=dtype)
    AV = np.empty_like(V)
    m, products, restarts = 0, 0, 0
    pending = []
    initializing = True

    def independent(vector, image=None):
        vector = np.array(vector, dtype=np.result_type(V.dtype, vector), copy=True)
        image = None if image is None else np.array(image, dtype=np.result_type(AV.dtype, image), copy=True)
        original = np.linalg.norm(vector)
        if not original:
            return None, None
        for _ in range(2):
            coefficients = V[:, :m].conj().T @ vector
            vector -= V[:, :m] @ coefficients
            if image is not None:
                image -= AV[:, :m] @ coefficients
        norm = np.linalg.norm(vector)
        if norm <= 1e-12*original:
            return None, None
        return vector/norm, None if image is None else image/norm

    def append(vector, image=None):
        nonlocal m, V, AV, products
        if image is None and initializing:
            pending.append(m)
            image = np.zeros(n,dtype=V.dtype)
        elif image is None:
            image = np.asarray(matmat(vector[:, None]))
            if image.shape != (n, 1):
                raise ValueError('Davidson operator returned an invalid block')
            image = image[:, 0]
            products += 1
        if not np.all(np.isfinite(image)):
            raise ValueError('Davidson operator returned nonfinite data')
        promoted = np.result_type(V, image)
        if promoted != V.dtype:
            V, AV = V.astype(promoted), AV.astype(promoted)
        V[:, m], AV[:, m] = vector, image
        m += 1

    target = min(limit, n, max(roots+3, 2*roots+1))
    for vector, image in zip(initial, known):
        if m >= min(target, limit-1 if limit > roots else limit):
            break
        vector, image = independent(vector, image)
        if vector is not None:
            append(vector, image)
    for index in np.argsort(diagonal, kind='stable'):
        if m >= max(roots, target-1):
            break
        vector = np.zeros(n)
        vector[index] = 1.
        vector, _ = independent(vector)
        if vector is not None:
            append(vector)
    # A dense guard reduces the risk of remaining confined to a guessed
    # symmetry block. As in any iterative eigensolver, it is not a certificate
    # that all lower disconnected states have been found.
    if m < target:
        vector, _ = independent(rng.normal(size=n))
        if vector is not None:
            append(vector)
    initializing = False
    if pending:
        block = np.asarray(matmat(V[:,pending]))
        if block.shape != (n,len(pending)) or not np.all(np.isfinite(block)):
            raise ValueError('Davidson operator returned an invalid initial block')
        promoted = np.result_type(V,block)
        V,AV = V.astype(promoted,copy=False),AV.astype(promoted,copy=False)
        AV[:,pending] = block
        products += len(pending)
    for iteration in range(1, maxiter+1):
        reduced = V[:, :m].conj().T @ AV[:, :m]
        if not np.allclose(reduced, reduced.conj().T, atol=1e-10, rtol=1e-10):
            raise ValueError('Davidson operator/history is not Hermitian')
        values, coefficients = np.linalg.eigh((reduced+reduced.conj().T)/2)
        energies = values[:roots]
        vectors = V[:, :m] @ coefficients[:, :roots]
        images = AV[:, :m] @ coefficients[:, :roots]
        residuals = images-vectors*energies
        norms = np.linalg.norm(residuals, axis=0)
        if callback is not None:
            callback(dict(iteration=iteration, energies=energies.tolist(), residual_norms=norms.tolist(),
                          dimension=m, products=products, restarts=restarts, subspace_limit=limit))
        if np.all(norms <= tolerance):
            audited = np.asarray((audit or matmat)(vectors))
            products += roots
            if audited.shape != vectors.shape or not np.all(np.isfinite(audited)):
                raise ValueError('Davidson audit returned an invalid block')
            norms = np.linalg.norm(audited-vectors*energies, axis=0)
            if np.any(norms > tolerance):
                raise RuntimeError('Excitation eigensolve did not converge (true residual)')
            return DavidsonEigenResult(energies, vectors, norms, iteration, products, restarts, limit)
        if iteration == maxiter:
            break
        if m+min(roots, np.count_nonzero(norms > tolerance)) > limit:
            keep = min(m, max(roots, min(2*roots, limit-1)))
            new_v = V[:, :m] @ coefficients[:, :keep]
            new_av = AV[:, :m] @ coefficients[:, :keep]
            V[:, :keep], AV[:, :keep] = new_v, new_av
            m = keep
            restarts += 1
        # Orthogonalize a whole correction block before applying the operator.
        start = m
        for root in np.argsort(-norms):
            if norms[root] <= tolerance or m == limit:
                continue
            denominator = diagonal-energies[root]
            denominator = np.where(abs(denominator) < denominator_floor,
                                   np.where(denominator < 0, -denominator_floor, denominator_floor), denominator)
            vector, _ = independent(-residuals[:, root]/denominator)
            if vector is None:
                vector, _ = independent(residuals[:, root])
            if vector is not None:
                V[:, m] = vector
                m += 1
        if m == start and m < limit:
            vector, _ = independent(rng.normal(size=n))
            if vector is not None:
                V[:, m] = vector
                m += 1
        if m == start:
            raise RuntimeError('Excitation Davidson stagnated before convergence')
        block = np.asarray(matmat(V[:, start:m]))
        products += m-start
        if block.shape != (n, m-start) or not np.all(np.isfinite(block)):
            raise ValueError('Davidson operator returned an invalid block')
        promoted = np.result_type(AV, block)
        if promoted != AV.dtype:
            V, AV = V.astype(promoted), AV.astype(promoted)
        AV[:, start:m] = block
    raise RuntimeError(f'Excitation Davidson did not converge after {maxiter} iterations (max residual={max(norms):.3e})')


def davidson_liu(hvp, hdiag, roots, tol=1e-3, *, verbose=False, **options):
    """Compatible Hessian interface to the bounded block Davidson solver."""
    callback = options.pop('callback',None)
    def report(event):
        if verbose:
            print(f"Davidson iteration {event['iteration']}: residuals={event['residual_norms']}", flush=True)
        if callback is not None:
            callback(event)
    result = davidson_eigenpairs(hvp, hdiag, roots, tolerance=tol, callback=report, **options)
    return result.energies, result.vectors


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
    if verbose:
        print('Not converged...')
    return x


def davidson_response(A, b, hdiag, history=None, omega=None, gamma=None,
                      tol=1e-6, verbose=False, *, maxiter=500,
                      denominator_floor=1e-10, callback=None, memory_bytes=128*1024**2, audit=None):
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
    ``callback`` receives each one-based iteration's residual norm, absolute
    tolerance and subspace dimension. With ``audit``, apparent convergence is
    checked with a fresh unshifted operator image. A failed audit replaces the
    cached residual and restarts from x and its true image before continuing;
    it never loosens the requested tolerance. Refresh events include
    ``residual_replacement=True``. The accepted audit can be retained by the
    caller, avoiding another expensive action after return.
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
    if not np.isfinite(memory_bytes) or memory_bytes <= 0:
        raise ValueError('Response Davidson memory budget must be positive')
    limit = min(n, 64)
    while limit and 16*(2*n*limit+4*limit*limit+8*n) > memory_bytes:
        limit -= 1
    if limit < min(n, 2):
        raise ValueError('Response Davidson memory budget cannot hold a correction subspace')

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
        old_v, old_av = old_v[:, :limit], old_av[:, :limit]
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

    if old_v.shape[1]:
        gram = old_v.conj().T @ old_v
        if not np.allclose(gram, np.eye(len(gram)), atol=1e-12, rtol=0):
            raise ValueError('Davidson history must contain orthonormal vectors')
        m = old_v.shape[1]
        V[:, :m], AV[:, :m] = old_v, old_av
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
        if callback is not None:
            callback(dict(iteration=iteration+1, residual_norm=float(residual_norm), tolerance=tol, dimension=m))
        if verbose:
            print(f'Iteration {iteration+1:3d} residual norm: {residual_norm:6e}')
        if residual_norm <= tol and audit is not None:
            ax = np.asarray(audit(x))
            if ax.shape != b.shape or not np.all(np.isfinite(ax)):
                raise ValueError('Response audit returned an invalid vector')
            residual = b-ax-shift*x
            residual_norm = scipy.linalg.norm(residual)
            if residual_norm > tol:
                # Long matrix-free actions can accumulate roundoff differently
                # on x than on the cached basis combination AV @ coefficients.
                # Re-anchor both V and AV to the fresh image before correcting.
                norm = scipy.linalg.norm(x)
                m = 0
                if norm:
                    append(x/norm,ax/norm)
                if callback is not None:
                    callback(dict(iteration=iteration+1,residual_norm=float(residual_norm),
                                  tolerance=tol,dimension=m,residual_replacement=True))
                # Resolve the refreshed reduced equation before generating a
                # new direction: the correction may already lie in span(x).
                continue
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
