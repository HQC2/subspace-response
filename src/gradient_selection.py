"""Batched ADAPT-style insertion gradients within the SD neighborhood.

Candidates are the same scalar fermionic blocks and SD occupation neighborhood
as SPQE. New rotations act on the current state (append in circuit order).
"""
from itertools import combinations
from math import comb

import numpy as np

from excitation_neighborhood import ExcitationNeighborhood
from excitations import singlet_excitations_for_block


def insertion_gradient(simulator, excitation, state, residual):
    """2 Re <(H-E)psi | dV/dtheta(0) psi>, before normalization.

    At zero angle the derivative of the scalar's factorized circuit is the
    sum of its monomial generators, independently of the Trotter step count.
    No Hamiltonian application or tangent state allocation is needed here.
    """
    value = 0.
    for term in excitation.terms:
        source, target, signs = simulator.pairs(term.holes, term.particles)
        value += term.coefficient*np.real(np.vdot(residual[target], signs*state[source])
                                         - np.vdot(residual[source], signs*state[target]))
    return float(2*value)


def _pack_monomials(simulator, monomials):
    """Immutable compact tables, reusable while only the wave function changes."""
    sources, targets, signs, a_starts, b_starts = [], [], [], [0], []
    for holes, particles in monomials:
        alpha, beta, mask, constant = simulator._excitation_layout(holes, particles)
        sources.extend((alpha[0], beta[0]))
        targets.extend((alpha[1], beta[1]))
        signs.extend((simulator.string_parities(mask >> 1)[alpha[0]]*(-1 if constant else 1),
                      simulator.string_parities(mask)[beta[0]]))
        b_starts.append(a_starts[-1]+len(alpha[0]))
        a_starts.append(b_starts[-1]+len(beta[0]))
    dtype = np.int32 if simulator.dimension <= np.iinfo(np.int32).max else np.int64
    sources = np.concatenate(sources).astype(dtype) if sources else np.empty(0, dtype=dtype)
    targets = np.concatenate(targets).astype(dtype) if targets else np.empty(0, dtype=dtype)
    signs = np.concatenate(signs) if signs else np.empty(0, dtype=np.int8)
    a_starts, b_starts = np.array(a_starts), np.array(b_starts, dtype=int)
    packed = a_starts, b_starts, sources, targets, signs
    for array in packed:
        array.flags.writeable = False
    return packed


def _contract_monomials(simulator, packed, state, residual, enabled):
    a_starts, b_starts, sources, targets, signs = packed
    output = np.zeros(len(b_starts))
    serial = parallel = None
    if simulator.kernel_backend != 'numpy':
        from uscc_kernels import insertion_gradients, parallel_insertion_gradients
        serial, parallel = insertion_gradients, parallel_insertion_gradients
    if serial is not None:
        simulator._run_numba_kernel(serial, parallel, output, state, residual,
                                   simulator.shape[1], enabled, a_starts, b_starts,
                                   sources, targets, signs,
                                   threads=simulator.effective_screening_threads())
    else:
        for i in range(len(output)):
            if not enabled[i]:
                continue
            a, b = slice(a_starts[i], b_starts[i]), slice(b_starts[i], a_starts[i+1])
            source = (sources[a, None]*simulator.shape[1]+sources[b]).ravel()
            target = (targets[a, None]*simulator.shape[1]+targets[b]).ravel()
            sign = (signs[a, None]*signs[b]).ravel()
            output[i] = 2*np.real(np.vdot(residual[target], sign*state[source])
                                  - np.vdot(residual[source], sign*state[target]))
    return output


def monomial_gradients(simulator, monomials, state, residual):
    """One-off contraction of unit monomials, without caching determinant pairs."""
    return _contract_monomials(simulator, _pack_monomials(simulator, monomials),
                               state, residual, np.ones(len(monomials), dtype=bool))


class GradientSelection(ExcitationNeighborhood):
    """Exact insertion gradients, with a conservative bound before spin coupling."""
    def __init__(self, simulator, constraints):
        hf = int(simulator.indices[np.argmax(simulator.initial)])
        super().__init__(hf, simulator.n_modes, constraints)
        self.simulator = simulator
        self._planned_blocks = set()
        self._batches = []

    def _monomials(self, block):
        occupations = self._occupations(block)
        opens = [i for i, n in enumerate(occupations) if n == 1]
        base = sum(3 << (self.n_modes-2*i-2) for i, n in enumerate(occupations) if n == 2)
        base |= sum(1 << (self.n_modes-2*i-2) for i in opens)
        def modes(bits):
            result = []
            while bits:
                mode = self.n_modes-bits.bit_length()
                result.append(mode)
                bits ^= 1 << (self.n_modes-mode-1)
            return tuple(result)
        for alpha in combinations(opens, len(opens)//2):
            index = base ^ sum(3 << (self.n_modes-2*i-2) for i in alpha)
            yield modes(self.hf_index & ~index), modes(index & ~self.hf_index)

    def select(self, state, applied, excitations, epsilon):
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError('Selection threshold must be finite and nonnegative')
        self._update_candidates(excitations)
        present = {e.id for e in excitations}
        present_blocks = {e.block for e in excitations}
        norm = np.vdot(state, state).real
        energy = float(np.vdot(state, applied).real/norm)
        residual = applied-energy*state
        floor = 64*np.finfo(float).eps*max(1., abs(energy), np.linalg.norm(applied)/np.sqrt(norm))
        proposals, candidates, screened, evaluated = [], 0, 0, 0
        max_gradient = max_bound = 0.

        def score_batch(blocks, monomials, starts, counts, packed):
            nonlocal candidates, screened, evaluated, max_gradient, max_bound
            enabled = np.ones(len(monomials), dtype=bool)
            missing_counts = list(counts)
            for i, block in enumerate(blocks):
                if block in present_blocks:
                    missing_counts[i] = sum(e.id not in present for e in self._blocks[block])
                if not missing_counts[i]:
                    enabled[starts[i]:starts[i+1]] = False
            derivatives = _contract_monomials(self.simulator, packed, state, residual, enabled)/norm
            for i, block in enumerate(blocks):
                count = missing_counts[i]
                if not count:
                    continue
                keys = monomials[starts[i]:starts[i+1]]
                values = derivatives[starts[i]:starts[i+1]]
                # Scalar coefficient vectors have unit Euclidean norm. By
                # Cauchy-Schwarz every scalar gradient is bounded by ||g_m||.
                # Pad for coefficient normalization and floating-point error.
                bound = float(np.linalg.norm(values))*(1+1e-10)
                max_bound = max(max_bound, bound)
                candidates += count
                if bound < epsilon or bound <= floor:
                    screened += 1
                    continue
                if block not in self._blocks:
                    self._blocks[block] = singlet_excitations_for_block(*block)
                by_term = dict(zip(keys, values))
                score = 0.
                for excitation in self._blocks[block]:
                    if excitation.id not in present:
                        evaluated += 1
                        value = abs(sum(term.coefficient*by_term[term.holes, term.particles]
                                        for term in excitation.terms))
                        score = max(score, value)
                max_gradient = max(max_gradient, score)
                if score > floor and score >= epsilon:
                    proposals.append((block, score, 'gradient'))

        # Cache only circuit-independent structure, never derivative values.
        # Replacing the ansatz can shrink the neighborhood, invalidating plans.
        if not self._planned_blocks <= self._candidates:
            self._batches.clear()
            self._planned_blocks.clear()
        blocks, monomials, starts, counts = [], [], [0], []
        def retain_batch():
            self._batches.append((tuple(blocks), tuple(monomials), tuple(starts),
                                  tuple(counts), _pack_monomials(self.simulator, monomials)))
        for block in sorted(self._candidates-self._planned_blocks):
            blocks.append(block)
            monomials.extend(self._monomials(block))
            starts.append(len(monomials))
            # Singlet multiplicity of the open-shell spins gives the pool
            # count without constructing the scalar coefficient vectors.
            opens = sum(n == 1 for n in self._occupations(block))
            counts.append(comb(opens, opens//2)-comb(opens, opens//2-1) if opens else 1)
            if len(monomials) >= 4096:
                retain_batch()
                blocks, monomials, starts, counts = [], [], [0], []
        if blocks:
            retain_batch()
        self._planned_blocks.update(self._candidates)
        for block in present_blocks & self._candidates:
            if block not in self._blocks:
                self._blocks[block] = singlet_excitations_for_block(*block)
        for batch in self._batches:
            score_batch(*batch)
        proposals.sort(key=lambda item: (-item[1], item[0]))
        return proposals, dict(candidate_blocks=len(self._candidates),
                               candidate_operators=candidates, source_blocks=len(self._sources),
                               screened_blocks=screened, evaluated_operators=evaluated,
                               max_evaluated_gradient=max_gradient,
                               max_candidate_gradient_bound=max_bound)
