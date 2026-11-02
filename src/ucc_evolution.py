"""Shared definitions of the UCC evolution and internal product formulas."""

SPIN_ADAPTATIONS = ('trotter', 'scalar', 'exact')


def validate_evolution(spin_adaptation='trotter', trotter_order=1, steps=1):
    if spin_adaptation not in SPIN_ADAPTATIONS:
        raise ValueError('spin_adaptation must be trotter, scalar or exact')
    if type(trotter_order) is not int or trotter_order not in (1, 2):
        raise ValueError('trotter_order must be 1 or 2')
    if type(steps) is not int or steps < 1:
        raise ValueError('trotter_steps must be a positive integer')


def component_schedule(excitation, steps=1, order=1):
    """Yield (term, scale) in application order within one scalar factor."""
    validate_evolution(trotter_order=order, steps=steps)
    terms = excitation.terms
    for _ in range(steps):
        if order == 1:
            yield from ((term, term.coefficient/steps) for term in terms)
        else:
            # Fuse the central pair of half steps. A one-term block stays exact.
            yield from ((term, term.coefficient/(2*steps)) for term in terms[:-1])
            yield terms[-1], terms[-1].coefficient/steps
            yield from ((term, term.coefficient/(2*steps)) for term in reversed(terms[:-1]))
