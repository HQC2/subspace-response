"""Optional compiled determinant-pair rotations for the sector simulator.

The packed plan preserves every component and Trotter step in its original
order. No fast-math reassociation or gate reordering is used.
"""
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _forward(parameters, state, parameter_ids, scales, starts, stops, sources, targets, signs):
    for gate in range(len(parameter_ids)):
        angle = parameters[parameter_ids[gate]]*scales[gate]
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


forward = None if njit is None else njit(cache=True)(_forward)
reverse = None if njit is None else njit(cache=True)(_reverse)
