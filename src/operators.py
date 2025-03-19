import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import RZ, RX, CNOT, Hadamard


class iHSingleExcitation(Operation):
    num_wires = AnyWires
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
    num_wires = AnyWires
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
