"""Sparse first-order SPQE residuals, H|HF> + [H,A]|HF>.

The amplitudes are physical scalar-generator coefficients. No full wave function,
sector vector, transformed Hamiltonian, or dense excitation pool is constructed.
Slater--Condon connections are evaluated on the sparse support of A|HF>.
Disconnected forward doubles cancel exactly against A H|HF>; forward singles
use the difference between the determinant and reference Fock elements.
"""
from math import comb
import numpy as np

from excitation_neighborhood import ExcitationNeighborhood
from excitations import apply_fermion_word, singlet_excitations_for_block
from spqe import _select_scores

try:
    from numba import njit
except ImportError:
    def njit(**kwargs):
        return lambda function: function


@njit(cache=True)
def _parity(bits):
    bits ^= bits >> 32
    bits ^= bits >> 16
    bits ^= bits >> 8
    bits ^= bits >> 4
    bits ^= bits >> 2
    bits ^= bits >> 1
    return 1-2*(bits & 1)


@njit(cache=True)
def _move(index, holes, particles, bits):
    sign = 1
    for i in holes:
        sign *= _parity(index >> (len(bits)-i))
        index ^= bits[i]
    for a in particles[::-1]:
        sign *= _parity(index >> (len(bits)-a))
        index ^= bits[a]
    return index, sign


@njit(cache=True)
def _sparse_residual(indices, amplitudes, h, g, hf, electrons, irreps, max_rank):
    modes = len(h)
    bits = np.array([1 << (modes-i-1) for i in range(modes)], dtype=np.int64)
    fock = h.copy()
    for a in range(modes):
        for i in range(modes):
            for j in range(electrons):
                fock[a,i] += g[a,j,i,j]
    reference_energy = 0.
    for i in range(electrons):
        reference_energy += h[i,i]
        for j in range(electrons):
            reference_energy += .5*g[i,j,i,j]
    # Seed the dictionary's types without allocating the fixed-electron sector.
    output = {hf: 0.}
    reference_doubles = {hf: 0.}
    for i in range(electrons):
        for a in range(electrons, modes):
            if i % 2 == a % 2 and irreps[i//2] == irreps[a//2]:
                target, sign = _move(hf, (i,), (a,), bits)
                if fock[a,i] != 0:
                    output[target] = sign*fock[a,i]
        for j in range(i+1, electrons):
            for a in range(electrons, modes):
                for b in range(a+1, modes):
                    if (i % 2+j % 2 != a % 2+b % 2 or
                            irreps[i//2] ^ irreps[j//2] ^ irreps[a//2] ^ irreps[b//2]):
                        continue
                    if g[a,b,i,j] != 0:
                        target, sign = _move(hf, (i,j), (a,b), bits)
                        reference_doubles[target] = sign*g[a,b,i,j]
                        if max_rank >= 2:
                            output[target] = sign*g[a,b,i,j]
    for column in range(len(indices)):
        amplitude = amplitudes[column]
        if amplitude == 0:
            continue
        index = indices[column]
        occupied = [i for i in range(modes) if index & bits[i]]
        virtual = [a for a in range(modes) if not index & bits[a]]
        rank = 0
        for i in occupied:
            rank += i >= electrons
        diagonal = -reference_energy
        for i in occupied:
            diagonal += h[i,i]
            for j in occupied:
                diagonal += .5*g[i,j,i,j]
        output[index] = output.get(index, 0.)+amplitude*diagonal
        for ii in range(len(occupied)):
            i = occupied[ii]
            for a in virtual:
                if i % 2 != a % 2 or irreps[i//2] != irreps[a//2]:
                    continue
                target_rank = rank+(a >= electrons)-(i >= electrons)
                if target_rank == 0 or target_rank > max_rank:
                    continue
                value = h[a,i]
                for j in occupied:
                    value += g[a,j,i,j]
                if i < electrons <= a:
                    value -= fock[a,i]  # -T H|HF>, including noncanonical HF.
                if value != 0:
                    target, sign = _move(index, (i,), (a,), bits)
                    output[target] = output.get(target, 0.)+amplitude*sign*value
            for jj in range(ii+1, len(occupied)):
                j = occupied[jj]
                for aa in range(len(virtual)):
                    a = virtual[aa]
                    for bb in range(aa+1, len(virtual)):
                        b = virtual[bb]
                        # Disjoint pure forward doubles commute with T.
                        if i < electrons and j < electrons and a >= electrons and b >= electrons:
                            continue
                        if (i % 2+j % 2 != a % 2+b % 2 or
                                irreps[i//2] ^ irreps[j//2] ^ irreps[a//2] ^ irreps[b//2]):
                            continue
                        target_rank = rank+(a >= electrons)+(b >= electrons)-(i >= electrons)-(j >= electrons)
                        if target_rank == 0 or target_rank > max_rank or g[a,b,i,j] == 0:
                            continue
                        target, sign = _move(index, (i,j), (a,b), bits)
                        output[target] = output.get(target, 0.)+amplitude*sign*g[a,b,i,j]
        # +T^dagger H|HF> has non-reference support only for selected singles.
        if rank == 1:
            hole, particle = 0, electrons
            for i in range(electrons):
                if not index & bits[i]:
                    hole = i
            for a in range(electrons,modes):
                if index & bits[a]:
                    particle = a
            _, reference_sign = _move(hf, (hole,), (particle,), bits)
            for source, value in reference_doubles.items():
                if source & bits[particle] and not source & bits[hole]:
                    target, sign = _move(source, (particle,), (hole,), bits)
                    output[target] = output.get(target, 0.)+amplitude*reference_sign*sign*value
    del output[hf]
    return np.array(list(output.keys()), dtype=np.int64), np.array(list(output.values()))


class LinearSPQESelection(ExcitationNeighborhood):
    """First-order residual screening with SPQE's cumulative block L2 budget."""
    def __init__(self, integrals, constraints):
        modes, electrons = integrals.qubits, integrals.electrons
        if modes > 62:
            raise ValueError('Sparse first-order SPQE currently supports up to 62 spin orbitals')
        hf = ((1 << electrons)-1) << (modes-electrons)
        super().__init__(hf, modes, constraints)
        self.integrals = integrals
        self.h, v = integrals.spin_integrals()
        antisym = v-v.swapaxes(0,1)
        self.g = .5*(antisym.swapaxes(2,3)-antisym)
        self.irreps = np.asarray(constraints.orbital_irreps or (0,)*self.n_orbitals, dtype=np.int64)
        self._references = {}
        self._occupation_blocks = {}

    def generators(self, block):
        if block not in self._blocks:
            self._blocks[block] = singlet_excitations_for_block(*block)
        return self._blocks[block]

    def residual(self, excitations, parameters):
        """Sparse computational-basis coefficients; the HF entry is omitted."""
        if np.iscomplexobj(parameters) and np.any(np.imag(parameters)):
            raise ValueError('Expected real generator amplitudes')
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (len(excitations),) or not np.all(np.isfinite(parameters)):
            raise ValueError('Expected one finite real amplitude per excitation')
        coefficients = {}
        for excitation, theta in zip(excitations, parameters):
            for term in excitation.terms:
                key = term.holes, term.particles
                if key not in self._references:
                    self._references[key] = apply_fermion_word(self.hf_index, self.n_modes, term.word)
                index, sign = self._references[key]
                coefficients[index] = coefficients.get(index, 0.)+theta*term.coefficient*sign
        indices = np.fromiter(coefficients, dtype=np.int64)
        amplitudes = np.fromiter(coefficients.values(), dtype=float)
        indices, values = _sparse_residual(indices, amplitudes, self.h, self.g, self.hf_index,
                                         self.integrals.electrons, self.irreps, self.constraints.max_rank)
        return dict(zip(indices.tolist(), values.tolist()))

    def select(self, excitations, parameters, epsilon):
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError('Selection threshold must be finite and nonnegative')
        vector = self.residual(excitations, parameters)
        # Every surviving term is at most an SD move from its source (or HF).
        # Generate only nonzero support, rather than enumerating zero neighbors.
        self._sources = {((), ()), *(e.block for e in excitations)}
        present = {e.id for e in excitations}
        present_counts = {}
        for e in excitations:
            present_counts[e.block] = present_counts.get(e.block,0)+1
        indices = np.fromiter(vector, dtype=np.int64)
        values = np.fromiter(vector.values(), dtype=float)
        beta_mask = sum(1 << (2*i) for i in range(self.n_orbitals))
        occupations = (indices & beta_mask)+((indices >> 1) & beta_mask)
        codes, inverse = np.unique(occupations, return_inverse=True)
        norms = np.sqrt(np.bincount(inverse, weights=values**2))
        self._candidates = set()
        scores = []
        for code, norm in zip(codes,norms):
            if norm == 0:
                continue
            if code not in self._occupation_blocks:
                numbers = tuple((int(code) >> (self.n_modes-2*i-2)) & 3
                                for i in range(self.n_orbitals))
                block = self._occupation_block(numbers)
                opens = numbers.count(1)
                count = comb(opens,opens//2)-comb(opens,opens//2+1) if opens else 1
                self._occupation_blocks[code] = block, count
            block, count = self._occupation_blocks[code]
            if not block[0] or not self.constraints.allowed(block):
                continue
            self._candidates.add(block)
            if present_counts.get(block,0) == count:
                continue
            if block in present_counts:
                # Avoid subtracting nearly equal squared norms for a partially
                # selected block: project its missing scalar rows explicitly.
                missing = []
                for excitation in self.generators(block):
                    if excitation.id in present:
                        continue
                    value = 0.
                    for term in excitation.terms:
                        key = term.holes, term.particles
                        if key not in self._references:
                            self._references[key] = apply_fermion_word(self.hf_index, self.n_modes, term.word)
                        index, sign = self._references[key]
                        value += term.coefficient*sign*vector.get(index, 0.)
                    missing.append(value)
                norm = np.linalg.norm(missing)
            # H and A are spin scalars. Their complete singlet block has the
            # same norm in the determinant and orthonormal CSF bases (Parseval).
            norm = float(norm)
            if norm:
                scores.append((block, norm, 'first_order_commutator'))
        proposals, diagnostics = _select_scores(scores, epsilon)
        diagnostics.update(candidate_blocks=len(self._candidates), source_blocks=len(self._sources),
                           residual_determinants=len(vector))
        return proposals, diagnostics
