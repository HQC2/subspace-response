"""Rank-independent factorized UCC, with Trotterized scalar excitation blocks."""
import pennylane as qml
from operators import FermionicExcitation


def apply_ucc(parameters, excitations, wires, trotter_steps=1):
    """Apply U only; the caller owns HF or arbitrary-state initialization.

    Each scalar excitation has one parameter shared by its ordered components.
    Repetitions split each group's angle; they do not repeat the outer ansatz.
    """
    if not isinstance(trotter_steps, int) or trotter_steps < 1:
        raise ValueError('trotter_steps must be a positive integer')
    if qml.math.shape(parameters) != (len(excitations),):
        raise ValueError('One parameter is required per scalar excitation')
    for theta, excitation in zip(parameters, excitations):
        for _ in range(trotter_steps):
            for term in excitation.terms:
                FermionicExcitation(theta*term.coefficient/trotter_steps, term.holes, term.particles, wires=wires)
