"""Single/double spatial occupation neighborhood shared by adaptive selectors."""
from itertools import combinations_with_replacement


class ExcitationNeighborhood:
    """HF and selected blocks supply sources, independently of their amplitudes."""
    def __init__(self, hf_index, n_modes, constraints):
        self.hf_index, self.n_modes, self.constraints = hf_index, n_modes, constraints
        self.n_orbitals = n_modes//2
        self.reference_occupations = tuple((hf_index >> (n_modes-2*i-2) & 3).bit_count()
                                           for i in range(self.n_orbitals))
        self._sources = set()
        self._candidates = set()
        self._blocks = {}
        # Base-three occupation codes let overlapping source neighborhoods be
        # deduplicated before constructing spatial blocks or checking symmetry.
        self._occupation_powers = tuple(3**i for i in range(self.n_orbitals))

    def _occupations(self, block):
        occupations = list(self.reference_occupations)
        for hole in block[0]:
            occupations[hole] -= 1
        for particle in block[1]:
            occupations[particle] += 1
        return occupations

    def _occupation_block(self, occupations):
        holes, particles = [], []
        for i, (reference, occupied) in enumerate(zip(self.reference_occupations, occupations)):
            holes.extend([i]*max(0, reference-occupied))
            particles.extend([i]*max(0, occupied-reference))
        return tuple(holes), tuple(particles)

    def _neighbors(self, source):
        occupations = self._occupations(source)
        holes = [i for i, n in enumerate(occupations) if n]
        particles = [i for i, n in enumerate(occupations) if n < 2]
        for rank in (1, 2):
            removals = ([(i,) for i in holes] if rank == 1 else
                        [pair for pair in combinations_with_replacement(holes, 2)
                         if pair[0] != pair[1] or occupations[pair[0]] == 2])
            additions = ([(a,) for a in particles] if rank == 1 else
                         [pair for pair in combinations_with_replacement(particles, 2)
                          if pair[0] != pair[1] or occupations[pair[0]] == 0])
            for removed in removals:
                for added in additions:
                    if any(i in added for i in removed):
                        continue  # Canceling spatial moves are already singles.
                    target = occupations.copy()
                    for i in removed:
                        target[i] -= 1
                    for a in added:
                        target[a] += 1
                    block = self._occupation_block(target)
                    if block[0] and self.constraints.allowed(block):
                        yield block

    def _update_candidates(self, excitations):
        sources = {((), ()), *(e.block for e in excitations)}
        if not self._sources <= sources:
            self._sources.clear()
            self._candidates.clear()
        targets = set()
        for source in sorted(sources-self._sources):
            occupations = self._occupations(source)
            code = sum(n*p for n, p in zip(occupations, self._occupation_powers))
            for rank in (1, 2):
                removals = self._occupation_moves(occupations, rank, removing=True)
                additions = self._occupation_moves(occupations, rank, removing=False)
                for removed_mask, removed_code in removals:
                    base = code-removed_code
                    targets.update(base+added_code for added_mask, added_code in additions
                                   if not removed_mask & added_mask)
            # Retain missing scalar couplings of an already selected block.
            if source[0] and self.constraints.allowed(source):
                self._candidates.add(source)
        for code in targets:
            occupations = []
            for _ in range(self.n_orbitals):
                code, n = divmod(code, 3)
                occupations.append(n)
            block = self._occupation_block(occupations)
            if block[0] and block not in self._candidates and self.constraints.allowed(block):
                self._candidates.add(block)
        self._sources = sources

    def _occupation_moves(self, occupations, rank, *, removing):
        available = [i for i, n in enumerate(occupations) if (n > 0 if removing else n < 2)]
        moves = ((i,) for i in available) if rank == 1 else (
            pair for pair in combinations_with_replacement(available, 2)
            if pair[0] != pair[1] or occupations[pair[0]] == (2 if removing else 0))
        return [(sum(1 << i for i in set(move)), sum(self._occupation_powers[i] for i in move))
                for move in moves]

    def generators(self, block):
        """Reuse discovered blocks even after eviction from the shared LRU."""
        return self._blocks[block]
