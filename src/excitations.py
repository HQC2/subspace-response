import numpy as np

def spin_adapted_excitations(electrons, qubits):
    nocc = electrons // 2
    nbas = qubits // 2
    # form spatial excitation singles, doubles
    space_singles = []
    space_doubles = []
    for i in range(nocc):
        for a in range(nocc, nbas):
            space_singles.append([i,a])
            for j in range(i, nocc):
                for b in range(a, nbas):
                    space_doubles.append([i,j,a,b])
    # form spin-orbital excitations
    singles = []
    idx_parameter = 0
    idx_excitation = 0
    parameter_map = {}
    for k, single in enumerate(space_singles):
        i, a = single
        aa = [2*i, 2*a] 
        bb = [2*i+1, 2*a+1]

        w = 1.0

        singles.append(aa)
        parameter_map[idx_excitation] = [[idx_parameter], [w]]
        idx_excitation += 1

        singles.append(bb)
        parameter_map[idx_excitation] = [[idx_parameter], [w]]
        idx_excitation += 1
        idx_parameter += 1

    doubles = []
    for k, double in enumerate(space_doubles):
        i, j, a, b = double
        aaaa = [2*i, 2*j, 2*a, 2*b]
        abab = [2*i, 2*j+1, 2*a, 2*b+1]
        baba = [2*i+1, 2*j, 2*a+1, 2*b]
        bbbb = [2*i+1, 2*j+1, 2*a+1, 2*b+1]
        abba = [2*i, 2*j+1, 2*a+1, 2*b]
        baab = [2*i+1, 2*j, 2*a, 2*b+1]

        
        if i==j and a == b:
            #abab
            doubles.append(abab)
            parameter_map[idx_excitation] = [[idx_parameter], [np.sqrt(2)]]
            idx_excitation += 1
            idx_parameter += 1
        elif i==j:
            # => occupied in same spatial, so first index is ab
            #abab + abba
            doubles.append(abab)
            parameter_map[idx_excitation] = [[idx_parameter], [1]]
            idx_excitation += 1
            doubles.append(abba)
            parameter_map[idx_excitation] = [[idx_parameter], [-1]]
            idx_excitation += 1
            idx_parameter += 1
        elif a==b:
            #abab + baab
            doubles.append(abab)
            parameter_map[idx_excitation] = [[idx_parameter], [1]]
            idx_excitation += 1

            doubles.append(baab)
            parameter_map[idx_excitation] = [[idx_parameter], [-1]]
            idx_excitation += 1
            idx_parameter += 1
        else:
            # R(1)
            wR1 = 0.5 * np.sqrt(2)
            doubles.append(abab)
            parameter_map[idx_excitation] = [[idx_parameter], [wR1]]
            idx_excitation += 1
            doubles.append(baba)
            parameter_map[idx_excitation] = [[idx_parameter], [wR1]]
            idx_excitation += 1
            doubles.append(baab)
            parameter_map[idx_excitation] = [[idx_parameter], [-wR1]]
            idx_excitation += 1
            doubles.append(abba)
            parameter_map[idx_excitation] = [[idx_parameter], [-wR1]]
            idx_excitation += 1
            idx_parameter += 1

            # R(2)
            wR2 = 1/(2*np.sqrt(3)) * np.sqrt(2)
            doubles.append(aaaa)
            parameter_map[idx_excitation] = [[idx_parameter], [2*wR2]]
            idx_excitation += 1
            doubles.append(bbbb)
            parameter_map[idx_excitation] = [[idx_parameter], [2*wR2]]
            idx_excitation += 1
            doubles.append(abab)
            parameter_map[idx_excitation] = [[idx_parameter], [wR2]]
            idx_excitation += 1
            doubles.append(baba)
            parameter_map[idx_excitation] = [[idx_parameter], [wR2]]
            idx_excitation += 1
            doubles.append(baab)
            parameter_map[idx_excitation] = [[idx_parameter], [wR2]]
            idx_excitation += 1
            doubles.append(abba)
            parameter_map[idx_excitation] = [[idx_parameter], [wR2]]
            idx_excitation += 1
            idx_parameter += 1
    return singles, doubles, parameter_map
