from functools import lru_cache

import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from pennylane.ops import RZ, RX, CNOT, Hadamard
from excitations import ExcitationTerm


class iHSingleExcitation(Operation):
    num_wires = None
    grad_method = "A"
    parameter_frequencies = [(0.5, 1.0)]

    def __init__(self, weight, wires=None, id=None):
        if len(wires) < 2:
            raise ValueError(f"expected at least two wires; got {len(wires)}")

        shape = qml.math.shape(weight)
        if shape != ():
            raise ValueError(f"Weight must be a scalar tensor {()}; got shape {shape}.")

        super().__init__(weight, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(weight, wires):  # pylint: disable=arguments-differ
        op_list = []
        op_list.append(qml.PauliRot(-weight/2, 'X'+ 'Z'*(len(wires)-2) +'X', wires=wires))
        op_list.append(qml.PauliRot(-weight/2, 'Y'+ 'Z'*(len(wires)-2) +'Y', wires=wires))

        return op_list

class iHDoubleExcitation(Operation):
    num_wires = None
    grad_method = "A"
    parameter_frequencies = [(0.5, 1.0)]

    def __init__(self, weight, wires1=None, wires2=None, id=None):
        if len(wires1) < 2:
            raise ValueError(
                f"expected at least two wires representing the occupied orbitals; "
                f"got {len(wires1)}"
            )
        if len(wires2) < 2:
            raise ValueError(
                f"expected at least two wires representing the unoccupied orbitals; "
                f"got {len(wires2)}"
            )

        shape = qml.math.shape(weight)
        if shape != ():
            raise ValueError(f"Weight must be a scalar; got shape {shape}.")

        wires1 = qml.wires.Wires(wires1)
        wires2 = qml.wires.Wires(wires2)

        self._hyperparameters = {
            "wires1": wires1,
            "wires2": wires2,
        }

        wires = wires1 + wires2
        super().__init__(weight, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(weight, wires, wires1, wires2):
        op_list = []

        op_list.append(qml.PauliRot( weight/8,  'Y'+'Z'*(len(wires1)-2)+'XY'+'Z'*(len(wires2)-2)+'X', wires=wires1+wires2))
        op_list.append(qml.PauliRot( weight/8,  'Y'+'Z'*(len(wires1)-2)+'XX'+'Z'*(len(wires2)-2)+'Y', wires=wires1+wires2))
        op_list.append(qml.PauliRot( weight/8,  'Y'+'Z'*(len(wires1)-2)+'YY'+'Z'*(len(wires2)-2)+'Y', wires=wires1+wires2))
        op_list.append(qml.PauliRot(-weight/8,  'Y'+'Z'*(len(wires1)-2)+'YX'+'Z'*(len(wires2)-2)+'X', wires=wires1+wires2))
        op_list.append(qml.PauliRot(-weight/8,  'X'+'Z'*(len(wires1)-2)+'XY'+'Z'*(len(wires2)-2)+'Y', wires=wires1+wires2))
        op_list.append(qml.PauliRot( weight/8,  'X'+'Z'*(len(wires1)-2)+'XX'+'Z'*(len(wires2)-2)+'X', wires=wires1+wires2))
        op_list.append(qml.PauliRot( weight/8,  'X'+'Z'*(len(wires1)-2)+'YY'+'Z'*(len(wires2)-2)+'X', wires=wires1+wires2))
        op_list.append(qml.PauliRot( weight/8,  'X'+'Z'*(len(wires1)-2)+'YX'+'Z'*(len(wires2)-2)+'Y', wires=wires1+wires2))

        return op_list


@lru_cache(maxsize=4096)
def excitation_pauli_terms(holes, particles, n_modes):
    """JW expansion of i(tau-tau†), checked to have commuting real terms."""
    term = ExcitationTerm(holes, particles)
    if max(holes + particles) >= n_modes:
        raise ValueError('Excitation mode outside the supplied wire register')
    word = term.fermi_word()
    sentence = qml.jordan_wigner(1j*(word-word.adjoint()), ps=True)
    terms = []
    for pauli, coefficient in sentence.items():
        if abs(complex(coefficient).imag) > 1e-12:
            raise ValueError('Expected a Hermitian excitation generator')
        if abs(coefficient) > 1e-14:
            terms.append((float(np.real(coefficient)), ''.join(pauli.get(i, 'I') for i in range(n_modes))))
    terms.sort(key=lambda x: x[1])
    for _, left in terms:
        for _, right in terms:
            if sum(a != b and a != 'I' and b != 'I' for a, b in zip(left, right)) % 2:
                raise ValueError('Noncommuting expansion of a pure excitation')
    return tuple(terms)


class FermionicExcitation(Operation):
    """exp(theta (tau-tau†)), with tau in ExcitationTerm canonical order.

    Holes/particles are mode positions in the full ordered ``wires`` register.
    This convention includes all Jordan-Wigner parity wires between endpoints.
    """
    num_wires = None
    grad_method = None

    def __init__(self, theta, holes, particles, wires=None, id=None):
        term = ExcitationTerm(tuple(holes), tuple(particles))
        if qml.math.shape(theta) != ():
            raise ValueError('Excitation angle must be scalar')
        if wires is None:
            wires = range(max(term.holes + term.particles) + 1)
        wires = qml.wires.Wires(wires)
        excitation_pauli_terms(term.holes, term.particles, len(wires))
        self._hyperparameters = dict(holes=term.holes, particles=term.particles)
        super().__init__(theta, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(theta, wires, holes, particles):
        return [qml.PauliRot(2*coefficient*theta, word, wires=wires) for coefficient, word in excitation_pauli_terms(holes, particles, len(wires))]


class FermionicTripleExcitation(FermionicExcitation):
    def __init__(self, theta, holes, particles, wires=None, id=None):
        if len(holes) != 3:
            raise ValueError('A triple excitation requires three holes')
        super().__init__(theta, holes, particles, wires=wires, id=id)


class FermionicQuadrupleExcitation(FermionicExcitation):
    def __init__(self, theta, holes, particles, wires=None, id=None):
        if len(holes) != 4:
            raise ValueError('A quadruple excitation requires four holes')
        super().__init__(theta, holes, particles, wires=wires, id=id)
