"""Rank-independent factorized UCC with optional exact scalar factors."""
from functools import lru_cache
import numpy as np
import pennylane as qml
from operators import FermionicExcitation
from ucc_evolution import component_schedule, validate_evolution


@lru_cache(maxsize=8)
def _scalar_spectrum(excitation, n_modes):
    """Dense full-register oracle; production scalar evolution uses the sector."""
    tau = excitation.sparse_matrix(n_modes)
    return np.linalg.eigh((1j*(tau-tau.T)).toarray())


def apply_ucc(parameters, excitations, wires, trotter_steps=1, *,
              spin_adaptation='trotter', trotter_order=1):
    """Apply U only; the caller owns HF or arbitrary-state initialization.

    Each scalar excitation has one parameter shared by its ordered components.
    Repetitions split each group's angle; they do not repeat the outer ansatz.
    """
    validate_evolution(spin_adaptation, trotter_order, trotter_steps)
    if spin_adaptation == 'exact':
        raise ValueError('spin_adaptation=exact requires the sector backend')
    if qml.math.shape(parameters) != (len(excitations),):
        raise ValueError('One parameter is required per scalar excitation')
    wires = qml.wires.Wires(wires)
    for theta, excitation in zip(parameters, excitations):
        if spin_adaptation == 'scalar':
            eigenvalues, eigenvectors = _scalar_spectrum(excitation, len(wires))
            qml.QubitUnitary(eigenvectors.conj().T, wires=wires)
            # Decompose the diagonal into differentiable RZ/MultiRZ operations
            # so Lightning's adjoint differentiation sees scalar parameters.
            qml.DiagonalQubitUnitary.compute_decomposition(
                qml.math.exp(-1j*theta*eigenvalues), wires=wires)
            qml.QubitUnitary(eigenvectors, wires=wires)
        else:
            for term, scale in component_schedule(excitation, trotter_steps, trotter_order):
                FermionicExcitation(theta*scale, term.holes, term.particles, wires=wires)
