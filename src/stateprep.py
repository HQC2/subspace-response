#!/usr/bin/env python

# Implemented by Oskar Graulund Lentz Rasmussen
# implementation following
# 10.1109/DAC18074.2021.9586240
from math import atan2
import random
import scipy
import numpy as np
import pennylane as qml

# Utility functions, modify S in-place
def XGate(S, qubit):
    for i in range(len(S)):
        stateAsList = list(S[i])
        if stateAsList[qubit] == '1':
            stateAsList[qubit] = '0'
        else:
            stateAsList[qubit] = '1'
        S[i] = ''.join(stateAsList)

def CXGate(S, control, target):
    for i in range(len(S)):
        stateAsList = list(S[i])
        if stateAsList[control] == '1' and stateAsList[target] == '0':
            stateAsList[target] = '1'
        elif stateAsList[control] == '1' and stateAsList[target] == '1':
            stateAsList[target] = '0'
        S[i] = ''.join(stateAsList)

def Algorithm1(S, coeffs):
    noq = len(S[0])

    Scopy = S.copy()

    # Make dif_qubits and dif_values
    T = Scopy.copy()
    dif_qubits = []
    dif_values = []
    while len(T) > 1:
        maxDiff, qubitForMaxDiff = -1, 0
        for qubit in range(0, noq):
            zeroCount, oneCount = 0, 0
            for state in T:
                if (state[qubit] == '0'):
                    zeroCount += 1
                else:
                    oneCount += 1

            if (abs(zeroCount - oneCount) > maxDiff and zeroCount != 0 and oneCount != 0):
                maxDiff = abs(zeroCount - oneCount)
                qubitForMaxDiff = qubit
    
        dif_qubits.append(qubitForMaxDiff)

        T0, T1 = [], []
        for state in T:
            if (state[qubitForMaxDiff] == '0'):
                T0.append(state)
            else:
                T1.append(state)
        
        if len(T0) < len(T1):
            T = T0.copy()
            dif_values.append(0)
        else:
            T = T1.copy()
            dif_values.append(1)

    dif = dif_qubits.pop()
    dif_values.pop()
    x1 = T[0]

    # Make T'
    Tprime = []
    for state in S:
        allQubitValuesMatch = True
        for i in range(0, len(dif_qubits)):
            if int(state[dif_qubits[i]]) != dif_values[i]:
                allQubitValuesMatch = False
                break
        
        if allQubitValuesMatch == True and state != x1:
            Tprime.append(state)
    
    while len(Tprime) > 1:
        maxDiff, qubitNumberForMaxDiff = -1, 0
        for qubit in range(0, noq):
            zeroCount = 0
            oneCount = 0
            for state in Tprime:
                if (state[qubit] == '0'):
                    zeroCount += 1
                else:
                    oneCount += 1

            if (abs(zeroCount - oneCount) > maxDiff and zeroCount != 0 and oneCount != 0):
                maxDiff = abs(zeroCount - oneCount)
                qubitNumberForMaxDiff = qubit
    
        dif_qubits.append(qubitNumberForMaxDiff)

        T0, T1 = [], []
        for state in Tprime:
            if (state[qubitNumberForMaxDiff] == '0'):
                T0.append(state)
            else:
                T1.append(state)
        
        if len(T0) < len(T1):
            Tprime = T0.copy()
            dif_values.append(0)
        else:
            Tprime = T1.copy()
            dif_values.append(1)

    # Change states
    x2 = Tprime[0]
    if int(x1[dif]) == 0:
        qml.X(dif)
        XGate(Scopy, dif)
    
    for i in range(0, noq):
        if i != dif and x1[i] != x2[i]:
            qml.CNOT([dif, i])
            CXGate(Scopy, dif, i)

    for qubit in dif_qubits:
        if int(x2[qubit]) == 0:
            qml.X(qubit)
            XGate(Scopy, qubit)

    # x1 is |1>. x2 is |0>
    # idx1 is the index of x1 in S. (similar for idx2)
    idx1, idx2 = 0, 0
    for i in range(len(S)):
        if S[i] == x1:
            idx1 = i
        if S[i] == x2:
            idx2 = i
            
    cx1, cx2 = coeffs[idx1], coeffs[idx2]
    rotationAngle = -2 * atan2(cx1, cx2)

    if len(dif_qubits) != 0:
        c = np.cos(rotationAngle/2)
        s = np.sin(rotationAngle/2)
        U = np.array([[c,-s], [s, c]])
        qml.ControlledQubitUnitary(U, wires=dif_qubits + [dif])
    elif len(dif_qubits) == 0:
        qml.RY(rotationAngle, dif)

    Sprime = [Scopy[idx2]]
    newCoeffs = [np.sqrt(cx1 * cx1 + cx2 * cx2)]
    for i in range(0, len(Scopy)):
        if i != idx1 and i != idx2:
            Sprime.append(Scopy[i])
            newCoeffs.append(coeffs[i])

    return Sprime, newCoeffs

def Algorithm2(inputStates, inputCoeffs):
    noq = len(inputStates[0])
    coeffs = inputCoeffs.copy()
    S = inputStates.copy()

    while len(S) > 1:
        S, coeffs = Algorithm1(S, coeffs)

    for i in range(0, noq):
        if S[0][i] == '1':
            qml.X(i)

def stateprep(statevector):
    if not scipy.sparse.issparse(statevector):
        raise ValueError(f'Only sparse state vectors are supported. Found {type(statevector)=}')
    if not (1 in statevector.shape):
        raise ValueError(f'Only one statevector (row or column) allowed, {statevector.shape=}.')
    if not len(statevector.indptr) == 2:
        raise ValueError(f'Bad choice of csr/csc array. Use a (N,1) csc or a (1,N) csr array.')
    norm = np.linalg.norm(statevector.data)
    if not np.isclose(norm, 1):
        raise ValueError(f'Invalid normalization {norm=}')
    num_qubits = int(np.log2(statevector.shape[0]*statevector.shape[1]))
    qml.adjoint(Algorithm2)([f'{index:0{num_qubits}b}' for index in statevector.indices], list(statevector.data))
