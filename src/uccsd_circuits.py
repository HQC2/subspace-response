# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

r"""
Contains the UCCSD template.
Added spin adaptation stuff.
Based on pennylane/templates/subroutines/uccsd.py
"""
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import BasisState

class UCCSD(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(
        self, weights, wires, excitations_ground_state, init_state
    ):
        shape = qml.math.shape(weights)
        init_state = qml.math.toarray(init_state)

        if init_state.dtype != np.dtype("int"):
            raise ValueError(f"Elements of 'init_state' must be integers; got {init_state.dtype}")

        self._hyperparameters = {"init_state": init_state, "excitations_ground_state": excitations_ground_state}
        super().__init__(weights, wires=wires)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(
        weights, wires, excitations_ground_state, init_state
    ):  # pylint: disable=arguments-differ
        op_list = []

        op_list.append(BasisState(init_state, wires=wires))

        for i, (excitations, excitation_weights) in enumerate(excitations_ground_state):
            for excitation, excitation_weight in zip(excitations, excitation_weights):
                if len(excitation) == 2:
                    r, p = excitation
                    s_wires = list(range(r, p + 1))
                    op_list.append(qml.FermionicSingleExcitation(weights[i]*excitation_weight, wires=s_wires))
                elif len(excitation) == 4:
                    s, r, q, p = excitation
                    w1 = list(range(s, r + 1))
                    w2 = list(range(q, p + 1))
                    op_list.append(qml.FermionicDoubleExcitation(weights[i]*excitation_weight, wires1=w1, wires2=w2))
                else:
                    raise ValueError
        return op_list

class UCCSD_exc(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(
        self, weights_ground_state, weights_excitation, wires, excitations_ground_state, init_state,
        excitations_singlet=None, excitations_triplet=None,
    ):
        if init_state.dtype != np.dtype("int"):
            raise ValueError(f"Elements of 'init_state' must be integers; got {init_state.dtype}")

        self._hyperparameters = {"init_state": init_state, 
                                 "excitations_ground_state": excitations_ground_state,
                                 "excitations_singlet": excitations_singlet,
                                 "excitations_triplet": excitations_triplet}

        super().__init__(weights_ground_state, weights_excitation, wires=wires)

    @property
    def num_params(self):
        return 2

    @staticmethod
    def compute_decomposition(
        weights_ground_state, weights_excitation, wires, excitations_ground_state, init_state,
        excitations_singlet=None, excitations_triplet=None,
    ):  # pylint: disable=arguments-differ
        op_list = []
        op_list.append(BasisState(init_state, wires=wires))

        # excitation things
        excitations = []
        if excitations_singlet is not None:
            excitations += excitations_singlet
        if excitations_triplet is not None:
            excitations += excitations_triplet
        assert len(excitations) == len(weights_excitation)
        for i, (excitations, excitation_weights) in enumerate(excitations):
            for excitation, excitation_weight in zip(excitations, excitation_weights):
                if len(excitation) == 2:
                    r, p = excitation
                    s_wires = list(range(r, p + 1))
                    op_list.append(qml.FermionicSingleExcitation(weights_excitation[i]*excitation_weight, wires=s_wires))
                elif len(excitation) == 4:
                    s, r, q, p = excitation
                    w1 = list(range(s, r + 1))
                    w2 = list(range(q, p + 1))
                    op_list.append(qml.FermionicDoubleExcitation(weights_excitation[i]*excitation_weight, wires1=w1, wires2=w2))
                else:
                    raise ValueError

        # ground-state
        for i, (excitations, excitation_weights) in enumerate(excitations_ground_state):
            for excitation, excitation_weight in zip(excitations, excitation_weights):
                if len(excitation) == 2:
                    r, p = excitation
                    s_wires = list(range(r, p + 1))
                    op_list.append(qml.FermionicSingleExcitation(weights_ground_state[i]*excitation_weight, wires=s_wires))
                elif len(excitation) == 4:
                    s, r, q, p = excitation
                    w1 = list(range(s, r + 1))
                    w2 = list(range(q, p + 1))
                    op_list.append(qml.FermionicDoubleExcitation(weights_ground_state[i]*excitation_weight, wires1=w1, wires2=w2))
                else:
                    raise ValueError
        return op_list
