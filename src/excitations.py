from dataclasses import dataclass
from functools import cached_property, lru_cache
from hashlib import sha256
from itertools import combinations_with_replacement, permutations, product
import json

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
    statevec = scipy.sparse.lil_array((2**num_qubits, 1))
    statevec[index] = 1
    return statevec.tocsc()

def excitation_to_statevector(ref_state, excitations, weights):
    num_qubits = len(ref_state)
    statevec = scipy.sparse.lil_array((2**num_qubits, 1))
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
    return -statevec.tocsc()


# Arbitrary-rank ground-state operators. The original SD routines above retain
# their historical signs/normalizations for existing response calculations.
def apply_fermion_word(index, n_modes, word):
    """Apply a written left-to-right sequence of (mode, creation) operators.

    Mode zero is the most significant bit. Return (index, phase), or None for
    Pauli blocking. Parities use the intermediate occupation after each action.
    """
    if not isinstance(n_modes, (int, np.integer)) or n_modes < 1 or not isinstance(index, (int, np.integer)) or index < 0 or index >= 2**n_modes:
        raise ValueError('Invalid occupation index or register size')
    index = int(index)
    phase = 1
    for mode, creation in reversed(tuple(word)):
        if not 0 <= mode < n_modes:
            raise ValueError('Fermion mode outside the register')
        bit = 1 << (n_modes - mode - 1)
        occupied = bool(index & bit)
        if occupied == creation:
            return None
        phase *= (-1) ** ((index >> (n_modes - mode)).bit_count())
        index ^= bit
    return index, phase


@dataclass(frozen=True)
class ExcitationTerm:
    """c a†[p0] ... a†[pr-1] a[hr-1] ... a[h0], with sorted indices."""
    holes: tuple
    particles: tuple
    coefficient: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, 'holes', tuple(self.holes))
        object.__setattr__(self, 'particles', tuple(self.particles))
        if not self.holes or len(self.holes) != len(self.particles):
            raise ValueError('An excitation needs equally many holes and particles')
        for indices in (self.holes, self.particles):
            if any(not isinstance(i, (int, np.integer)) or i < 0 for i in indices):
                raise ValueError('Mode indices must be nonnegative integers')
            if tuple(sorted(set(indices))) != indices:
                raise ValueError('Mode indices must be distinct and sorted')
        if set(self.holes) & set(self.particles):
            raise ValueError('Holes and particles must be disjoint')
        if not np.isfinite(self.coefficient) or not np.isreal(self.coefficient):
            raise ValueError('USCC coefficients must be finite and real')
        object.__setattr__(self, 'coefficient', float(self.coefficient))

    @property
    def rank(self):
        return len(self.holes)

    @property
    def word(self):
        return tuple((p, True) for p in self.particles) + tuple((h, False) for h in reversed(self.holes))

    def apply(self, index, n_modes):
        result = apply_fermion_word(index, n_modes, self.word)
        return None if result is None else (result[0], self.coefficient * result[1])

    def fermi_word(self):
        import pennylane as qml
        return qml.FermiWord({(i, mode): '+' if creation else '-' for i, (mode, creation) in enumerate(self.word)})

    def to_dict(self):
        return dict(holes=list(self.holes), particles=list(self.particles), coefficient=self.coefficient)


@dataclass(frozen=True)
class Excitation:
    """A normalized scalar excitation; component order defines its Trotter block."""
    spatial_holes: tuple
    spatial_particles: tuple
    coupling: int
    terms: tuple

    def __post_init__(self):
        object.__setattr__(self, 'spatial_holes', tuple(self.spatial_holes))
        object.__setattr__(self, 'spatial_particles', tuple(self.spatial_particles))
        object.__setattr__(self, 'terms', tuple(self.terms))
        if not self.terms or any(t.rank != self.rank for t in self.terms):
            raise ValueError('All components must have the specified excitation rank')
        keys = [(t.holes, t.particles) for t in self.terms]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate excitation components')
        for term in self.terms:
            if tuple(sorted(h // 2 for h in term.holes)) != self.spatial_holes or tuple(sorted(p // 2 for p in term.particles)) != self.spatial_particles:
                raise ValueError('Component does not match the spatial occupation block')
            if sum(h % 2 for h in term.holes) != sum(p % 2 for p in term.particles):
                raise ValueError('Ground-state components must conserve M_S')
        if not np.isclose(sum(t.coefficient**2 for t in self.terms), 1., atol=1e-12):
            raise ValueError('Excitation reference states must be normalized')

    @property
    def rank(self):
        return len(self.spatial_holes)

    @property
    def block(self):
        return self.spatial_holes, self.spatial_particles

    @cached_property
    def id(self):
        return sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:24]

    def to_dict(self):
        return dict(spatial_holes=list(self.spatial_holes), spatial_particles=list(self.spatial_particles), coupling=self.coupling, terms=[t.to_dict() for t in self.terms])

    @classmethod
    def from_dict(cls, data):
        return cls(tuple(data['spatial_holes']), tuple(data['spatial_particles']), data['coupling'], tuple(ExcitationTerm(**t) for t in data['terms']))

    def reference_state(self, ref_state):
        ref_state = np.asarray(ref_state)
        if ref_state.ndim != 1 or not np.all(np.isin(ref_state, [0,1])):
            raise ValueError('Reference occupations must be a one-dimensional bitstring')
        n_modes = len(ref_state)
        index = int(sum(int(bit) << (n_modes - i - 1) for i, bit in enumerate(ref_state)))
        entries = {}
        for term in self.terms:
            result = term.apply(index, n_modes)
            if result is not None:
                entries[result[0]] = entries.get(result[0], 0.) + result[1]
        rows = [i for i, c in entries.items() if abs(c) > 1e-14]
        return scipy.sparse.csc_array(([entries[i] for i in rows], (rows, np.zeros(len(rows), dtype=int))), shape=(2**n_modes, 1))

    def sparse_matrix(self, n_modes):
        """Small-system operator oracle; never needed by production screening."""
        rows, cols, values = [], [], []
        for index in range(2**n_modes):
            for term in self.terms:
                result = term.apply(index, n_modes)
                if result is not None:
                    rows.append(result[0]); cols.append(index); values.append(result[1])
        return scipy.sparse.csr_array((values, (rows, cols)), shape=(2**n_modes, 2**n_modes))


@lru_cache(maxsize=2048)
def singlet_excitations_for_block(spatial_holes, spatial_particles):
    """Expand spin-free E products and deterministically remove Gram null modes.

    All hole orbitals are doubly occupied and all particle orbitals empty in
    the reference. Modified Gram-Schmidt operates on canonical monomial
    coefficients: their reference states are orthogonal with unit norm.
    """
    holes, particles = tuple(spatial_holes), tuple(spatial_particles)
    if not holes or len(holes) != len(particles):
        raise ValueError('Invalid occupation block')
    if holes != tuple(sorted(holes)) or particles != tuple(sorted(particles)):
        raise ValueError('Spatial indices must be sorted')
    if any(i < 0 for i in holes + particles) or set(holes) & set(particles):
        raise ValueError('Invalid or overlapping spatial indices')
    if any(holes.count(i) > 2 for i in holes) or any(particles.count(i) > 2 for i in particles):
        return ()
    n_modes = 2 * (max(holes + particles) + 1)
    hf = sum((1 << (n_modes - 2*h - 1)) + (1 << (n_modes - 2*h - 2)) for h in set(holes))
    basis = []
    for pairing in sorted(set(permutations(particles))):
        raw = {}
        for spins in product((0, 1), repeat=len(holes)):
            word = tuple(op for h, p, spin in zip(holes, pairing, spins) for op in ((2*p+spin, True), (2*h+spin, False)))
            applied = apply_fermion_word(hf, n_modes, word)
            if applied is None:
                continue
            hs = tuple(sorted(2*h+spin for h, spin in zip(holes, spins)))
            ps = tuple(sorted(2*p+spin for p, spin in zip(pairing, spins)))
            canonical = ExcitationTerm(hs, ps).apply(hf, n_modes)
            key = hs, ps
            raw[key] = raw.get(key, 0.) + applied[1] / canonical[1]
        # Reorthogonalization stabilizes blocks with several spin couplings.
        for _ in range(2):
            for previous in basis:
                overlap = sum(previous.get(k, 0.) * c for k, c in raw.items())
                for k, c in previous.items():
                    raw[k] = raw.get(k, 0.) - overlap*c
        raw = {k: c for k, c in raw.items() if abs(c) > 1e-12}
        norm = np.sqrt(sum(c*c for c in raw.values()))
        if norm < 1e-10:
            continue
        sign = np.sign(raw[min(raw)])
        basis.append({k: c/(norm*sign) for k, c in sorted(raw.items())})
    return tuple(Excitation(holes, particles, i, tuple(ExcitationTerm(h, p, c) for (h, p), c in vector.items())) for i, vector in enumerate(basis))


def singlet_excitations(electrons, qubits, ranks=(1, 2, 3, 4), orbital_irreps=None):
    """Enumerate normalized singlet blocks for a closed-shell active reference.

    Optional Abelian orbital irrep IDs use the XOR multiplication convention.
    """
    if electrons < 0 or electrons % 2 or qubits % 2 or electrons > qubits:
        raise ValueError('A closed-shell reference requires even electron and mode counts')
    nocc, norb = electrons // 2, qubits // 2
    if orbital_irreps is not None and len(orbital_irreps) != norb:
        raise ValueError('One Abelian irrep ID is required per spatial orbital')
    result = []
    for rank in sorted(set(ranks)):
        if not isinstance(rank, (int, np.integer)) or rank < 1:
            raise ValueError('Ranks must be positive integers')
        if rank > min(electrons, qubits-electrons):
            continue
        hs = [h for h in combinations_with_replacement(range(nocc), rank) if all(h.count(i) <= 2 for i in h)]
        ps = [p for p in combinations_with_replacement(range(nocc, norb), rank) if all(p.count(i) <= 2 for i in p)]
        for holes in hs:
            for particles in ps:
                if orbital_irreps is not None:
                    irrep = 0
                    for i in holes + particles:
                        irrep ^= int(orbital_irreps[i])
                    if irrep:
                        continue
                result.extend(singlet_excitations_for_block(holes, particles))
    return result
