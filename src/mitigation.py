#!/usr/bin/env python


#!/usr/bin/env python

import functools
import pennylane as qml
from uccsd_circuits import UCCSD
import numpy as np

def bits(target_bitstring, device, ucc):
    @qml.qnode(device)
    def circuit():
        # we make "null" circuit with theta=0, which does nothing except accumulate noise
        # we need to use numerically large enough thetas so that gates are not pruned
        # we prepare the state from an empty HF reference, so no need to cancel with X gates after
        tiny_theta = np.zeros_like(ucc.theta)
        tiny_theta[:] = 1e-9
        UCCSD(tiny_theta, range(ucc.qubits), ucc.excitations_ground_state, ucc.hf_state*0)
        for i, bit in enumerate(target_bitstring):
            if bit:
                qml.X(i)
        return qml.probs()
    return circuit()

#@qml.qnode(dev)
def operator_probs(ucc, operator):
    UCCSD(ucc.theta, range(self.qubits), self.excitations_ground_state, self.hf_state)
    return qml.probs(op=operator)

def bin_array(num, m):
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

def get_confusion_matrix(ucc, device):
    measured = np.zeros((2**ucc.qubits, 2**ucc.qubits))
    for i in range(2**ucc.qubits):
        target_bitstring = bin_array(i, ucc.qubits)
        measured[:, i] = bits(target_bitstring, device, ucc)
    return measured

@qml.transform
def rem_mitigate(tape, confusion):
    probs_tape = tape.copy()
    probs_tape.measurements.pop(0)
    qubits = len(tape.wires)
#    I = functools.reduce(lambda a,b: a@b, [qml.Identity(i) for i in range(qubits)])
    for observable in tape.observables:
        for coeff, pauli in zip(*observable.terms()):
            probs_tape.measurements.append(qml.probs(op=pauli))

    def post_processing_fn(results):
        mitigated_expectation = 0.0
        for observable in tape.observables:
            for coeff, pauli, result in zip(*observable.terms(), results[0]):
                print(f'{coeff=}')
                print(f'{pauli=}')
                print(f'{result=}')
                if pauli == qml.Identity():
                    mitigated_expectation += coeff
                    continue
                trace_wires = tape.wires - pauli.wires
                trace_indices = tuple(list(trace_wires) + [qubits + w for w in trace_wires])
                N = 2**(len(tape.wires) - len(trace_wires))
                confusion_traced = confusion.reshape([2]*2*qubits).sum(axis=trace_indices).reshape(N,N) / 2**len(trace_wires)
                print(confusion_traced)
                print(result)
                mitigated_expectation += coeff*np.dot(np.linalg.solve(confusion_traced, result), pauli.eigvals())
        return mitigated_expectation

    return [probs_tape], post_processing_fn

