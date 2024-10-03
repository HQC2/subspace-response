#!/usr/bin/env python


#!/usr/bin/env python

import functools
import pennylane as qml
from uccsd_circuits import UCCSD
import numpy as np

def bits(target_bitstring, wires, device, ucc):
    @qml.qnode(device)
    def circuit():
        # we make "null" circuit with theta=0, which does nothing except accumulate noise
        # we prepare the state from an empty HF reference, so no need to cancel with X gates after
        UCCSD(np.zeros_like(ucc.theta), range(ucc.qubits), ucc.excitations_ground_state, ucc.hf_state*0)
        I = functools.reduce(lambda a,b: a@b, [qml.Identity(i) for i in wires])
        for i, bit in enumerate(target_bitstring):
            wire = wires[i]
            if bit:
                qml.X(wire)
        return qml.probs(op=I)
    return circuit()

def bin_array(num, m):
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

def get_confusion_matrix(ucc, device, wires=None):
    if not wires:
        wires = range(ucc.qubits)
    N = len(wires)
    measured = np.zeros((2**N, 2**N))
    I = functools.reduce(lambda a,b: a@b, [qml.Identity(i) for i in wires])
    for i in range(2**N):
        target_bitstring = bin_array(i, len(wires))
        measured[:, i] = bits(target_bitstring, wires, device, ucc)
    return measured

@qml.transform
def rem_mitigate(tape, confusion):
    probs_tape = tape.copy()
    print('measurements:', probs_tape.measurements)
    print('observables:', probs_tape.observables)
    while probs_tape.measurements:
        probs_tape.measurements.pop(0)
    print('measurements:', probs_tape.measurements)
    print('observables:', probs_tape.observables)
    qubits = len(tape.wires)
    pauli_ops = set()
    for observable in tape.observables:
        for coeff, pauli in zip(*observable.terms()):
            pauli_ops |= {pauli}
    pauli_ops = list(pauli_ops)
    for pauli in pauli_ops:
        probs_tape.measurements.append(qml.probs(op=pauli))
    
    def post_processing_fn(results):
        pauli_to_result = {}
        for pauli, result in zip(pauli_ops, results[0]):
            pauli_to_result[pauli] = result
        
        mitigated_expectation_values = []
        for observable in tape.observables:
            mitigated_expectation = 0.0
            for coeff, pauli in zip(*observable.terms()):
                result = pauli_to_result[pauli]
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
                mitigated_expectation += coeff*np.dot(np.linalg.inv(confusion_traced)@result, pauli.eigvals())
            mitigated_expectation_values.append(mitigated_expectation)
        if len(mitigated_expectation_values) == 1:
            # unpack for single expectation value
            mitigated_expectation_values = mitigated_expectation_values[0]
        return mitigated_expectation_values

    return [probs_tape], post_processing_fn

