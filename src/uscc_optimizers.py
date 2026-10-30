"""Diagonal-metric optimization for factorized USCC generator parameters.

The objective returns (energy, analytic gradient). Histories describe a local
inverse-Hessian approximation, not wave functions or response operators.
"""
import numpy as np
from scipy.optimize import OptimizeResult


def excitation_curvature(integrals, excitations):
    """Positive HF Fock estimate of the energy curvature: twice the gap."""
    # g[p,q,r,s]=(p s|q r), whereas F uses chemists' (p q|r s).
    eri = integrals.two_body.transpose(0, 3, 1, 2)
    eps = np.diag(integrals.one_body).copy()
    for i in range(integrals.electrons//2):
        eps += 2*np.diag(eri[:, :, i, i])-np.diag(eri[:, i, i, :])
    gaps = [sum(eps[p] for p in e.spatial_particles)-sum(eps[h] for h in e.spatial_holes) for e in excitations]
    return np.maximum(2*np.asarray(gaps), .1)


def _line_search(objective, x, energy, gradient, direction, max_steps=24):
    slope = float(gradient @ direction)
    if not np.isfinite(slope) or slope >= 0:
        return None
    noise = 64*np.finfo(float).eps*max(1., abs(energy))
    alpha = 1.
    for _ in range(max_steps):
        trial = x+alpha*direction
        if np.array_equal(trial, x):
            return None
        value, derivative = objective(trial)
        predicted = -alpha*slope
        armijo = (value <= energy+1e-4*alpha*slope
                  and (predicted > 10*noise or energy-value > noise))
        # Near stationarity the energy difference can be below floating-point
        # resolution even while the analytic gradient remains informative.
        roundoff = (predicted <= 10*noise and abs(value-energy) <= noise
                    and np.linalg.norm(derivative) <= (1-1e-4)*np.linalg.norm(gradient))
        if np.isfinite(value) and np.all(np.isfinite(derivative)) and (armijo or roundoff):
            return trial, float(value), np.asarray(derivative)
        alpha *= .5
    return None


def minimize_metric(objective, initial, curvature, *, gradient_tolerance=1e-6,
                    maxiter=300, history=None, algorithm='lbfgs', scale_metric=False,
                    memory=10, warmup=3):
    """Preconditioned L-BFGS or safeguarded gradient-descent/Pulay DIIS.

    Secant pairs can be padded with zeros by the caller when the ansatz grows.
    This embeds the learned old block and initializes new coordinates with the
    diagonal metric; new cross curvatures are learned by subsequent updates.
    Steps satisfy Armijo decrease or reduce the analytic gradient within the
    energy's roundoff resolution. Convergence requires the public L2 gradient
    target, rather than a small energy change.
    """
    x = np.array(initial, dtype=float, copy=True)
    curvature = np.asarray(curvature, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError('Expected finite initial parameters')
    if curvature.shape != x.shape or not np.all(np.isfinite(curvature)) or np.any(curvature <= 0):
        raise ValueError('Expected positive finite curvature matching parameters')
    if not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0 or type(maxiter) is not int or maxiter < 1:
        raise ValueError('Expected positive gradient tolerance and iteration limit')
    if algorithm not in ('lbfgs','diis') or type(memory) is not int or memory < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError('Invalid optimizer algorithm or history settings')
    inverse = 1/curvature
    energy, gradient = objective(x)
    gradient = np.asarray(gradient)
    if not np.isfinite(energy) or gradient.shape != x.shape or not np.all(np.isfinite(gradient)):
        raise ValueError('Objective returned invalid energy or gradient')
    pairs = [] if history is None else [(np.array(s, copy=True), np.array(y, copy=True)) for s, y in history]
    for s, y in pairs:
        if s.shape != x.shape or y.shape != x.shape or not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)) or s @ y <= 0:
            raise ValueError('Invalid curvature history')
    pairs = pairs[-memory:]
    points, errors = [], []
    searches, rejected = 0, 0
    for iteration in range(maxiter+1):
        if np.linalg.norm(gradient) <= gradient_tolerance:
            return OptimizeResult(x=x, fun=energy, jac=gradient, success=True, nit=iteration,
                                  message='L2 gradient converged', history=pairs, searches=searches, rejected=rejected)
        if iteration == maxiter:
            break
        diagonal_step = -inverse*gradient
        if algorithm == 'diis':
            points.append(x+diagonal_step)
            errors.append(np.sqrt(inverse)*gradient)
            points, errors = points[-memory:], errors[-memory:]
            direction = diagonal_step
            if iteration >= warmup:
                B = np.array(errors) @ np.array(errors).T
                scale = np.max(abs(B))
                augmented = np.ones((len(B)+1, len(B)+1))
                augmented[:-1, :-1] = B/scale if scale else B
                augmented[-1, -1] = 0.
                rhs = np.zeros(len(B)+1); rhs[-1] = 1.
                coefficients = np.linalg.lstsq(augmented, rhs, rcond=1e-12)[0][:-1]
                if np.max(abs(coefficients)) <= 20:
                    proposed = coefficients @ np.array(points)-x
                    if gradient @ proposed < 0:
                        direction = proposed
        elif algorithm == 'lbfgs':
            q = gradient.copy()
            alphas = []
            for s, y in reversed(pairs):
                alpha = (s @ q)/(s @ y)
                q -= alpha*y
                alphas.append(alpha)
            factor = 1.
            if scale_metric and pairs:
                s, y = pairs[-1]
                factor = np.clip((s @ y)/(y @ (inverse*y)), .05, 20.)
            r = factor*inverse*q
            for (s, y), alpha in zip(pairs, reversed(alphas)):
                r += s*(alpha-(y @ r)/(s @ y))
            direction = -r
        else:
            raise ValueError('algorithm must be lbfgs or diis')
        diagonal_direction = np.array_equal(direction, diagonal_step)
        step = _line_search(objective, x, energy, gradient, direction, max_steps=24 if diagonal_direction else 8)
        searches += 1
        if step is None:
            pairs.clear(); points.clear(); errors.clear()
            rejected += 1
            if not diagonal_direction:
                step = _line_search(objective, x, energy, gradient, diagonal_step)
                searches += 1
            if step is None:
                # Near a positive-curvature minimum the unscaled negative
                # gradient also decreases its norm, even when a poor diagonal
                # metric fails that roundoff-region merit test.
                step = _line_search(objective,x,energy,gradient,-gradient/np.max(curvature))
                searches += 1
        if step is None:
            return OptimizeResult(x=x, fun=energy, jac=gradient, success=False, nit=iteration,
                                  message='Line search failed', history=pairs, searches=searches, rejected=rejected)
        new_x, new_energy, new_gradient = step
        s, y = new_x-x, new_gradient-gradient
        sy = float(s @ y)
        if sy > 1e-10*np.linalg.norm(s)*np.linalg.norm(y) and sy > 0:
            pairs.append((s, y)); pairs = pairs[-memory:]
        x, energy, gradient = new_x, new_energy, new_gradient
    return OptimizeResult(x=x, fun=energy, jac=gradient, success=False, nit=maxiter,
                          message='Iteration limit reached', history=pairs, searches=searches, rejected=rejected)
