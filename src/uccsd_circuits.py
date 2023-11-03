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
        self, weights, wires, s_wires, d_wires, parameter_map, init_state, id=None
    ):
        if (not s_wires) and (not d_wires):
            raise ValueError(
                f"s_wires and d_wires lists can not be both empty; got ph={s_wires}, pphh={d_wires}"
            )

        for d_wires_ in d_wires:
            if len(d_wires_) != 2:
                raise ValueError(
                    f"expected entries of d_wires to be of size 2; got {d_wires_} of length {len(d_wires_)}"
                )

        shape = qml.math.shape(weights)
        init_state = qml.math.toarray(init_state)

        if init_state.dtype != np.dtype("int"):
            raise ValueError(f"Elements of 'init_state' must be integers; got {init_state.dtype}")

        self._hyperparameters = {"init_state": init_state, "s_wires": s_wires, "d_wires": d_wires,
                "parameter_map": parameter_map}

        super().__init__(weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(
        weights, wires, s_wires, d_wires, parameter_map, init_state
    ):  # pylint: disable=arguments-differ
        op_list = []

        op_list.append(BasisState(init_state, wires=wires))
        
        for i, (w1, w2) in enumerate(d_wires):
            weight = 0.0
            for idx, factor in zip(*parameter_map[len(s_wires) + i]):
                weight += factor * weights[idx]
            op_list.append(
                qml.FermionicDoubleExcitation(weight, wires1=w1, wires2=w2)
            )

        for i, s_wires_ in enumerate(s_wires):
            weight = 0.0
            for idx, factor in zip(*parameter_map[i]):
                weight += factor * weights[idx]
            op_list.append(qml.FermionicSingleExcitation(weight, wires=s_wires_))
        
        return op_list

class UCCSD_exc(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(
        self, weights_ground_state, weights_excitation, wires, s_wires, d_wires, parameter_map, init_state, id=None,
        s_wires_triplet=None, d_wires_triplet=None, parameter_map_triplet=None,
    ):
        if (not s_wires) and (not d_wires):
            raise ValueError(
                f"s_wires and d_wires lists can not be both empty; got ph={s_wires}, pphh={d_wires}"
            )

        for d_wires_ in d_wires:
            if len(d_wires_) != 2:
                raise ValueError(
                    f"expected entries of d_wires to be of size 2; got {d_wires_} of length {len(d_wires_)}"
                )

        init_state = qml.math.toarray(init_state)

        if init_state.dtype != np.dtype("int"):
            raise ValueError(f"Elements of 'init_state' must be integers; got {init_state.dtype}")

        self._hyperparameters = {"init_state": init_state, "s_wires": s_wires, "d_wires": d_wires,
                                 "parameter_map": parameter_map, "s_wires_triplet": s_wires_triplet,
                                 "d_wires_triplet": d_wires_triplet, "parameter_map_triplet": parameter_map_triplet}

        super().__init__(weights_ground_state, weights_excitation, wires=wires, id=id)

    @property
    def num_params(self):
        return 2

    @staticmethod
    def compute_decomposition(
        weights_ground_state, weights_excitation, wires, s_wires, d_wires, parameter_map, init_state,
        s_wires_triplet=None, d_wires_triplet=None, parameter_map_triplet=None,
    ):  # pylint: disable=arguments-differ
        op_list = []
        op_list.append(BasisState(init_state, wires=wires))
        

        if parameter_map_triplet is not None:
            for i, (w1, w2) in enumerate(d_wires_triplet):
                weight = 0.0
                for idx, factor in zip(*parameter_map_triplet[len(s_wires_triplet) + i]):
                    weight += factor * weights_excitation[idx]
                op_list.append(qml.FermionicDoubleExcitation(weight, wires1=w1, wires2=w2))
            for i, s_wires_ in enumerate(s_wires_triplet):
                weight = 0.0
                for idx, factor in zip(*parameter_map_triplet[i]):
                    weight += factor * weights_excitation[idx]
                op_list.append(qml.FermionicSingleExcitation(weight, wires=s_wires_))
        else:
            for i, (w1, w2) in enumerate(d_wires):
                weight = 0.0
                for idx, factor in zip(*parameter_map[len(s_wires) + i]):
                    weight += factor * weights_excitation[idx]
                op_list.append(qml.FermionicDoubleExcitation(weight, wires1=w1, wires2=w2))
            for i, s_wires_ in enumerate(s_wires):
                weight = 0.0
                for idx, factor in zip(*parameter_map[i]):
                    weight += factor * weights_excitation[idx]
                op_list.append(qml.FermionicSingleExcitation(weight, wires=s_wires_))

        for i, (w1, w2) in enumerate(d_wires):
            weight = 0.0
            for idx, factor in zip(*parameter_map[len(s_wires) + i]):
                weight += factor * weights_ground_state[idx]
            op_list.append(
                qml.FermionicDoubleExcitation(weight, wires1=w1, wires2=w2)
            )

        for i, s_wires_ in enumerate(s_wires):
            weight = 0.0
            for idx, factor in zip(*parameter_map[i]):
                weight += factor * weights_ground_state[idx]
            op_list.append(qml.FermionicSingleExcitation(weight, wires=s_wires_))
        return op_list
