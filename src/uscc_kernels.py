"""Optional compiled determinant rotations and scalar-basis orthogonalization.

The packed plan preserves every component and Trotter step in its original
order. No fast-math reassociation or gate reordering is used.
"""
import numpy as np

try:
    from numba import njit, prange
except ImportError:
    njit = None
    prange = range


def _forward(parameters, state, parameter_ids, scales, starts, stops, sources, targets, signs):
    for gate in range(len(parameter_ids)):
        angle = parameters[parameter_ids[gate]]*scales[gate]
        if angle == 0:
            continue
        cosine, sine = np.cos(angle), np.sin(angle)
        for pair in range(starts[gate],stops[gate]):
            source, target = sources[pair], targets[pair]
            left, right = state[source], state[target]
            signed_sine = signs[pair]*sine
            state[source] = cosine*left-signed_sine*right
            state[target] = signed_sine*left+cosine*right


def _reverse(parameters, state, costate, gradient, parameter_ids, scales, starts, stops, sources, targets, signs):
    for gate in range(len(parameter_ids)-1,-1,-1):
        parameter, scale = parameter_ids[gate], scales[gate]
        angle = parameters[parameter]*scale
        cosine, sine = np.cos(angle), np.sin(angle)
        derivative = 0.
        # Pairs within one monomial are disjoint. Accumulate its derivative and
        # undo state/costate rotations together, without gather temporaries.
        for pair in range(starts[gate],stops[gate]):
            source, target = sources[pair], targets[pair]
            left, right = state[source], state[target]
            cleft, cright = costate[source], costate[target]
            derivative += signs[pair]*(np.conj(cright)*left-np.conj(cleft)*right).real
            signed_sine = signs[pair]*sine
            state[source] = cosine*left+signed_sine*right
            state[target] = -signed_sine*left+cosine*right
            costate[source] = cosine*cleft+signed_sine*cright
            costate[target] = -signed_sine*cleft+cosine*cright
        gradient[parameter] += 2*scale*derivative


def _forward_parallel(parameters, state, parameter_ids, scales, starts, stops, sources, targets, signs):
    for gate in range(len(parameter_ids)):
        angle = parameters[parameter_ids[gate]]*scales[gate]
        if angle == 0:
            continue
        # Each gate is ordered; only its disjoint determinant pairs commute.
        # Small gates stay serial to avoid a barrier for too little work.
        if stops[gate]-starts[gate] < 2048:
            forward(parameters, state, parameter_ids[gate:gate+1], scales[gate:gate+1],
                    starts[gate:gate+1], stops[gate:gate+1], sources, targets, signs)
        else:
            cosine, sine = np.cos(angle), np.sin(angle)
            for pair in prange(starts[gate], stops[gate]):
                source, target = sources[pair], targets[pair]
                left, right = state[source], state[target]
                signed_sine = signs[pair]*sine
                state[source] = cosine*left-signed_sine*right
                state[target] = signed_sine*left+cosine*right


def _reverse_parallel(parameters, state, costate, gradient, parameter_ids, scales, starts, stops, sources, targets, signs):
    for gate in range(len(parameter_ids)-1, -1, -1):
        if stops[gate]-starts[gate] < 2048:
            reverse(parameters, state, costate, gradient, parameter_ids[gate:gate+1], scales[gate:gate+1],
                    starts[gate:gate+1], stops[gate:gate+1], sources, targets, signs)
        else:
            parameter, scale = parameter_ids[gate], scales[gate]
            angle = parameters[parameter]*scale
            cosine, sine = np.cos(angle), np.sin(angle)
            derivative = 0.
            for pair in prange(starts[gate], stops[gate]):
                source, target = sources[pair], targets[pair]
                left, right = state[source], state[target]
                cleft, cright = costate[source], costate[target]
                derivative += signs[pair]*(np.conj(cright)*left-np.conj(cleft)*right).real
                signed_sine = signs[pair]*sine
                state[source] = cosine*left+signed_sine*right
                state[target] = -signed_sine*left+cosine*right
                costate[source] = cosine*cleft+signed_sine*cright
                costate[target] = -signed_sine*cleft+cosine*cright
            gradient[parameter] += 2*scale*derivative


def _orthogonalize(raw, order, count, basis, columns, counts, size):
    """Two-pass MGS, retaining the original dictionary insertion/summation order.

    Dense storage makes the arithmetic compilable; order tracks exactly which
    components a dictionary would have inserted, including canceled entries.
    No fast-math reduction or reassociation is permitted here.
    """
    present = np.zeros(len(raw), dtype=np.bool_)
    for i in range(count):
        present[order[i]] = True
    for _ in range(2):
        for row in range(size):
            overlap = 0.
            for i in range(count):
                col = order[i]
                overlap += basis[row, col]*raw[col]
            for i in range(counts[row]):
                col = columns[row, i]
                if not present[col]:
                    present[col] = True
                    order[count] = col
                    count += 1
                raw[col] -= overlap*basis[row, col]
    norm = 0.
    for i in range(count):
        col = order[i]
        if abs(raw[col]) > 1e-12:
            norm += raw[col]*raw[col]
        else:
            raw[col] = 0.
    norm = np.sqrt(norm)
    if norm < 1e-10:
        return 0
    first = 0
    while raw[first] == 0:
        first += 1
    signed_norm = norm*(1. if raw[first] > 0 else -1.)
    count = 0
    for col in range(len(raw)):
        if raw[col] != 0:
            basis[size, col] = raw[col]/signed_norm
            columns[size, count] = col
            count += 1
    counts[size] = count
    return count


forward = None if njit is None else njit(cache=True)(_forward)
reverse = None if njit is None else njit(cache=True)(_reverse)
parallel_forward = None if njit is None else njit(cache=True, parallel=True)(_forward_parallel)
parallel_reverse = None if njit is None else njit(cache=True, parallel=True)(_reverse_parallel)
orthogonalize = None if njit is None else njit(cache=True)(_orthogonalize)
