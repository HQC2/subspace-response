"""Bit-mask acceleration of the original USCC max-product screening rule."""
import numpy as np
from uscc_kernels import njit


def _maximum_products(terms, amplitudes, channels, coefficients, max_rank):
    # Seed dictionary types explicitly for Numba, including an empty result.
    result = {(np.int64(0), np.int64(0)): 0.}
    ranks = {(np.int64(0), np.int64(0)): np.int64(0)}
    result.clear()
    ranks.clear()
    for i in range(len(amplitudes)):
        if amplitudes[i] == 0:
            continue
        for j in range(len(coefficients)):
            if (terms[i, 4]+channels[j, 4] > max_rank or
                    terms[i, 0] & channels[j, 0] or terms[i, 1] & channels[j, 1] or
                    terms[i, 5] ^ channels[j, 5]):
                continue
            block = terms[i, 2]+channels[j, 2], terms[i, 3]+channels[j, 3]
            value = abs(amplitudes[i]*coefficients[j])
            if value > result.get(block, -1.):
                result[block] = value
                ranks[block] = terms[i, 4]
    return result, ranks


maximum_products = _maximum_products if njit is None else njit(cache=True)(_maximum_products)


def product_scores(selector, amplitudes):
    """Return exactly the same candidates, maxima, and tie reasons as the oracle.

    Disjoint spin-orbital masks replace temporary Python sets. Two bits per
    spatial orbital encode occupation counts, so disjoint products can add
    their codes without constructing/sorting hole and particle tuples.
    """
    irreps = selector.orbital_irreps
    def layout(holes, particles):
        irrep = 0
        if irreps is not None:
            for i in holes+particles:
                irrep ^= irreps[i//2]
        return (sum(1 << i for i in holes), sum(1 << i for i in particles),
                sum(1 << (2*(i//2)) for i in holes),
                sum(1 << (2*(i//2)) for i in particles), len(holes), irrep)
    context = id(selector.channels), irreps
    cached = getattr(selector, '_product_channels', None)
    if cached is None or cached[0] != context:
        channels = [c for c in selector.channels if abs(c.coefficient) > 0]
        rows = np.array([layout(c.holes, c.particles) for c in channels], dtype=np.int64).reshape(-1, 6)
        coefficients = np.array([c.coefficient for c in channels])
        cached = selector._product_channels = context, rows, coefficients
    items = sorted(amplitudes.items())
    terms = np.array([layout(*key) for key, value in items], dtype=np.int64).reshape(-1, 6)
    values = np.array([value for key, value in items])
    result, ranks = maximum_products(terms, values, cached[1], cached[2], selector.max_rank)
    orbitals = selector.integrals.qubits//2
    def decode(code):
        return tuple(i for i in range(orbitals) for _ in range((int(code) >> (2*i)) & 3))
    scores = {}
    for (holes, particles), value in result.items():
        rank = ranks[holes, particles]
        block = decode(holes), decode(particles)
        scores[block] = float(value), f't{int(rank)}*h{len(block[0])-int(rank)}'
    return scores
