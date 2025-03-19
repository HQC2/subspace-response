import numpy as np
import scipy

def spin_adapted_excitations(electrons, qubits, triplet=False, generalized=False):
    nocc = electrons//2
    nbas = qubits//2
    # form spatial excitation singles, doubles
    space_singles = []
    space_doubles = []

    occ_start = 0
    occ_end = nocc
    vir_start = nocc
    vir_end = nbas
    if generalized:
        occ_end = nbas
        vir_start = 0

    for i in range(occ_start, occ_end):
        for a in range(max(i + 1, vir_start), vir_end):
            space_singles.append([i, a])
    for i in range(occ_start, occ_end):
        for j in range(i, occ_end):
            for a in range(max(i, j, vir_start), vir_end):
                for b in range(a, vir_end):
                    space_doubles.append([i, j, a, b])
    # form spin-orbital excitations
    excitations = []

    if not triplet:
        # singlet operators
        # doubles
        for k, double in enumerate(space_doubles):
            i, j, a, b = double
            aaaa = [2*i, 2*j, 2*a, 2*b]
            abab = [2*i, 2*j + 1, 2*a, 2*b + 1]
            baba = [2*i + 1, 2*j, 2*a + 1, 2*b]
            bbbb = [2*i + 1, 2*j + 1, 2*a + 1, 2*b + 1]
            abba = [2*i, 2*j + 1, 2*a + 1, 2*b]
            baab = [2*i + 1, 2*j, 2*a, 2*b + 1]

            if i == j and a == b:
                #abab
                excitations.append([[abab], [np.sqrt(2)/np.sqrt(2)]])
            elif i == j:
                # => occupied in same spatial, so first index is ab
                #abab + abba
                excitations.append([[abab, abba], [1/np.sqrt(2), -1/np.sqrt(2)]])
            elif a == b:
                #abab + baab
                excitations.append([[abab, baab], [1/np.sqrt(2), -1/np.sqrt(2)]])
            else:
                # R(1)
                wR1 = 0.5*np.sqrt(2)/np.sqrt(2)
                excitations.append([[abab, baba, baab, abba], [wR1, wR1, -wR1, -wR1]])
                # R(2)
                wR2 = 1/(2*np.sqrt(3))*np.sqrt(2)/np.sqrt(2)
                excitations.append([[aaaa, bbbb, abab, baba, baab, abba], [2*wR2, 2*wR2, wR2, wR2, wR2, wR2]])
        # singles
        for k, single in enumerate(space_singles):
            i, a = single
            aa = [2*i, 2*a]
            bb = [2*i + 1, 2*a + 1]
            excitations.append([[aa, bb], [1.0/np.sqrt(2), 1.0/np.sqrt(2)]])
    else:
        # triplet
        for k, double in enumerate(space_doubles):
            i, j, a, b = double
            aaaa = [2*i, 2*j, 2*a, 2*b]
            abab = [2*i, 2*j + 1, 2*a, 2*b + 1]
            baba = [2*i + 1, 2*j, 2*a + 1, 2*b]
            bbbb = [2*i + 1, 2*j + 1, 2*a + 1, 2*b + 1]
            abba = [2*i, 2*j + 1, 2*a + 1, 2*b]
            baab = [2*i + 1, 2*j, 2*a, 2*b + 1]
            if i == j and a == b:
                # no triplet possible
                continue
            elif i == j:
                # abab and abba
                excitations.append([[abab, abba], [1, 1]])
            elif a == b:
                # abab and baab
                excitations.append([[abab, baab], [1, 1]])
            else:
                # T(1)
                excitations.append([[aaaa, bbbb], [1, -1]])
                # T(2)
                wT2 = 0.5*np.sqrt(2)
                excitations.append([[abab, baba, abba, baab], [wT2, -wT2, -wT2, wT2]])
                # T(3)
                wT3 = 0.5*np.sqrt(2)
                excitations.append([[abab, baba, abba, baab], [wT2, -wT2, wT2, -wT2]])
        for k, single in enumerate(space_singles):
            i, a = single
            aa = [2*i, 2*a]
            bb = [2*i + 1, 2*a + 1]
            excitations.append([[aa, bb], [1, -1]])
    return excitations


def occupation_to_statevector(bitstring):
    # get statevector associated with a computational basis state 
    # bitstring should be an array of [0,1] integers, for example [1,1,0,0,0,0,0,0]
    num_qubits = len(bitstring)
    index = sum(bitstring * 2**np.arange(num_qubits-1, -1, -1))
    statevec = scipy.sparse.lil_matrix((2**num_qubits, 1))
    statevec[index] = 1
    return statevec

def excitation_to_statevector(ref_state, excitations, weights):
    num_qubits = len(ref_state)
    statevec = scipy.sparse.lil_matrix((2**num_qubits, 1))
    exc_vector = np.zeros_like(ref_state)
    for exc, w in zip(excitations, weights):
        # we flip bits according to exc
        # for example, if ref_state is [1,1,0,0,0,0,0,0]
        # and exc is                   [0,1,0,1,0,0,0,0]
        # we get                       [1,0,0,1,0,0,0,0]
        phase = 1
        r = len(exc)//2
        annihilation = exc[:r]
        creation = exc[r:]
        exc_vector[:] = ref_state
        for a in annihilation:
            if not ref_state[a]:
                continue
            phase *= (-1)**(ref_state[:a].sum())
            exc_vector[a] = 0
        for c in creation:
            if ref_state[c]:
                continue
            phase *= (-1)**(ref_state[:c].sum())
            exc_vector[c] = 1
        statevec += w * phase * occupation_to_statevector(exc_vector)
    return -statevec
