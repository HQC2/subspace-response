"""Exact scalar factors on small active-mode blocks with compressed spectators.

Move an excitation's active fermion modes ahead of the spectators. In that
ordering its generator is G_active tensor I. The permutation's fermionic
parity is retained in the embedding, so no Jordan-Wigner strings are dropped.
Only connected, nonzero blocks of G_active are diagonalized, once per coupling
template. A plan stores short alpha/beta string maps rather than sector-sized
matrices or one copy of each determinant address per spectator pair.
"""
from functools import lru_cache
from weakref import WeakValueDictionary

import numpy as np
from scipy import sparse
from scipy.linalg import schur
from scipy.sparse.csgraph import connected_components
from pyscf.fci import cistring


def _parities(indices, mask):
    parity = indices & mask
    for shift in (32, 16, 8, 4, 2, 1):
        parity ^= parity >> shift
    return (1-2*(parity & 1)).astype(np.int8)


class Spectrum:
    """Real Schur pairs reconstruct exp(tG) with quadratic retained storage."""
    def __init__(self, generator):
        self.generator = np.ascontiguousarray(generator)
        size = len(generator)
        if size == 2:
            self.frequencies = np.array([generator[0, 1]])
            self.basis = np.eye(2)
        else:
            t, q = schur(generator, output='real')
            frequencies, columns = [], []
            i = 0
            while i < size-1:
                if abs(t[i, i+1]) < 1e-13:
                    i += 1
                    continue
                a, b = q[:, i], q[:, i+1]
                frequencies.append(t[i, i+1])
                columns.extend((a, b))
                i += 2
            self.frequencies = np.array(frequencies)
            self.basis = np.ascontiguousarray(np.array(columns).reshape(-1, size).T)
        for a in (self.generator, self.frequencies, self.basis):
            a.flags.writeable = False

    def matrix(self, theta):
        angles = theta*self.frequencies
        # -2 sin²(x/2) retains accuracy of cos(x)-1 near zero.
        cosine, sine = -2*np.sin(angles/2)**2, np.sin(angles)
        a, b = self.basis[:, ::2], self.basis[:, 1::2]
        return (np.eye(len(self.generator))+(a*cosine) @ a.T+(b*cosine) @ b.T
                +(a*sine) @ b.T-(b*sine) @ a.T)


# Different coupling templates can contain the same small generator. Keep one
# immutable spectral decomposition while any live plan/template uses it, without
# retaining an unbounded archive of discarded plans.
_spectra = WeakValueDictionary()


@lru_cache(maxsize=128)
def local_blocks(terms):
    """Nonzero connected components on the active modes, shared across orbitals."""
    modes = 2*(max(max(h+p) for h, p, _ in terms)//2+1)
    rows, cols, values = [], [], []
    for holes, particles, coefficient in terms:
        hole_mask = sum(1 << (modes-h-1) for h in holes)
        flip = hole_mask | sum(1 << (modes-p-1) for p in particles)
        free = [i for i in range(modes) if not flip & (1 << i)]
        words = np.arange(1 << len(free), dtype=np.int64)
        sources = np.full(len(words), hole_mask, dtype=np.int64)
        for i, bit in enumerate(free):
            sources |= ((words >> i) & 1) << bit
        targets = sources ^ flip
        mask, toggled, constant = 0, 0, 0
        for mode in holes+tuple(reversed(particles)):
            prefix = ((1 << modes)-1) ^ ((1 << (modes-mode))-1)
            mask ^= prefix
            constant ^= (toggled & prefix).bit_count() & 1
            toggled ^= 1 << (modes-mode-1)
        weights = coefficient*_parities(sources, mask)*(-1 if constant else 1)
        rows.extend((targets, sources)); cols.extend((sources, targets))
        values.extend((weights, -weights))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    vertices = np.unique(rows)
    generator = sparse.coo_matrix((np.concatenate(values),
                                   (np.searchsorted(vertices, rows), np.searchsorted(vertices, cols))),
                                  shape=(len(vertices),)*2).tocsr()
    generator.eliminate_zeros()
    count, labels = connected_components(generator, directed=False)
    order = np.argsort(labels, kind='stable')
    boundaries = np.r_[0, np.cumsum(np.bincount(labels, minlength=count))]
    spectra, groups = {}, {}
    positions = np.empty(len(vertices), dtype=np.int64)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        members = order[start:stop]
        if len(members) < 2:
            continue
        # Connected components have no edges to other components. Read their
        # CSR rows directly instead of constructing two sparse slices per
        # tiny block (hundreds of thousands of slices in high-rank plans).
        if len(members) == 2:
            value = generator.data[generator.indptr[members[0]]]
            matrix = np.array([[0., value], [-value, 0.]])
        else:
            positions[members] = np.arange(len(members))
            matrix = np.zeros((len(members), len(members)))
            for row, member in enumerate(members):
                left, right = generator.indptr[member:member+2]
                matrix[row, positions[generator.indices[left:right]]] = generator.data[left:right]
        key = matrix.shape, matrix.tobytes()
        if key not in spectra:
            # Do not retain another dense matrix as a global dictionary key.
            # Verify hash hits so collisions can only reduce cache reuse.
            shared_key = matrix.shape, hash(key[1])
            spectrum = _spectra.get(shared_key)
            if spectrum is None or not np.array_equal(spectrum.generator, matrix):
                spectrum = Spectrum(matrix)
                _spectra[shared_key] = spectrum
            spectra[key] = spectrum
        states = vertices[members]
        alpha = sum(((states >> (modes-2*p-1)) & 1) << p for p in range(modes//2))
        beta = sum(((states >> (modes-2*p-2)) & 1) << p for p in range(modes//2))
        # Group equal matrices and local spin counts. Their spectators are
        # Cartesian copies of the same tiny transformation, so build all
        # string embeddings together instead of one Python object per copy.
        group = key, int(alpha[0]).bit_count(), int(beta[0]).bit_count()
        if group not in groups:
            groups[group] = spectra[key], [], []
        groups[group][1].append(alpha)
        groups[group][2].append(beta)
    return tuple((np.array(alpha), np.array(beta), spectrum)
                 for spectrum, alpha, beta in groups.values())


class Embedding:
    def __init__(self, spectrum, alpha, beta, alpha_signs, beta_signs):
        self.spectrum = spectrum
        self.alpha, self.beta = alpha, beta
        self.alpha_signs, self.beta_signs = alpha_signs, beta_signs
        self.copies = alpha.shape[0]*alpha.shape[1]*beta.shape[1]
        for a in (alpha, beta, alpha_signs, beta_signs):
            a.flags.writeable = False

    def addresses(self, width):
        return (self.alpha[:, :, None, :]*width+self.beta[:, None, :, :]).reshape(-1, self.alpha.shape[-1])

    def signs(self):
        return (self.alpha_signs[:, :, None, :]*self.beta_signs[:, None, :, :]).reshape(-1, self.alpha.shape[-1])


class EmbeddingGeometry:
    """Orbital/string maps shared by every spin coupling on these active modes."""
    def __init__(self, simulator, orbitals):
        self.simulator = simulator
        self.orbitals = orbitals
        self.spectators = sorted(set(range(simulator.n_orbitals))-set(orbitals))
        spectator_mask = sum(3 << (simulator.n_modes-2*p-2) for p in self.spectators)
        self.codes = np.arange(1 << len(orbitals), dtype=np.int64)
        self.counts = np.array([int(c).bit_count() for c in self.codes])
        self.masks = np.zeros(len(self.codes), dtype=np.int64)
        self.gauges = np.zeros(len(self.codes), dtype=np.int64)
        for i, p in enumerate(orbitals):
            occupied = (self.codes >> i) & 1
            self.masks |= occupied << p
            prefix = ((1 << simulator.n_modes)-1) ^ ((1 << (simulator.n_modes-2*p))-1)
            self.gauges ^= occupied*(prefix & spectator_mask)
        self._tables = {}
        self._embeddings = {}

    def tables(self, remaining):
        cached = self._tables.get(remaining)
        if cached is None:
            sim = self.simulator
            dtype = np.int32 if sim.dimension <= np.iinfo(np.int32).max else np.int64
            strings = cistring.make_strings(self.spectators, remaining)
            columns = np.zeros((len(self.codes), len(strings)), dtype=dtype)
            valid = np.flatnonzero(self.counts == sim.n_alpha-remaining)
            columns[valid] = np.searchsorted(sim._strings, self.masks[valid, None] | strings)
            bits = np.zeros(len(strings), dtype=np.int64)
            for p in self.spectators:
                bits |= ((strings >> p) & 1) << (sim.n_modes-2*p-2)
            phases = _parities(bits[None, :], self.gauges[:, None])
            columns.flags.writeable = phases.flags.writeable = False
            cached = self._tables[remaining] = columns, phases
        return cached

    def embedding(self, codes, remaining, *, phases=False):
        """Share immutable maps across scalar couplings on these orbitals."""
        key = phases, remaining, codes.shape, codes.tobytes()
        cached = self._embeddings.get(key)
        if cached is None:
            table = self.tables(remaining)[int(phases)]
            cached = np.ascontiguousarray(table[codes].transpose(0, 2, 1))
            cached.flags.writeable = False
            self._embeddings[key] = cached
        return cached


def excitation_plan(simulator, excitation, geometries):
    for term in excitation.terms:
        if max(term.holes+term.particles) >= simulator.n_modes:
            raise ValueError('Fermion mode outside the register')
        if sum(h % 2 for h in term.holes) != sum(p % 2 for p in term.particles):
            raise ValueError('Excitation leaves the fixed spin/electron sector')
    orbitals = tuple(sorted({mode//2 for t in excitation.terms for mode in t.holes+t.particles}))
    active = [2*p+s for p in orbitals for s in (0, 1)]
    addresses = {mode: i for i, mode in enumerate(active)}
    terms = tuple((tuple(addresses[h] for h in t.holes),
                   tuple(addresses[p] for p in t.particles), t.coefficient) for t in excitation.terms)
    geometry = geometries.get(orbitals)
    if geometry is None:
        geometry = geometries[orbitals] = EmbeddingGeometry(simulator, orbitals)
    result = []
    for alpha_codes, beta_codes, spectrum in local_blocks(terms):
        parity_codes = alpha_codes ^ beta_codes
        remaining_a = simulator.n_alpha-int(alpha_codes[0, 0]).bit_count()
        remaining_b = simulator.n_beta-int(beta_codes[0, 0]).bit_count()
        if not (0 <= remaining_a <= len(geometry.spectators)
                and 0 <= remaining_b <= len(geometry.spectators)):
            continue
        alpha = geometry.embedding(alpha_codes, remaining_a)
        beta = geometry.embedding(beta_codes, remaining_b)
        alpha_signs = geometry.embedding(parity_codes, remaining_a, phases=True)
        beta_signs = geometry.embedding(parity_codes, remaining_b, phases=True)
        result.append(Embedding(spectrum, alpha, beta, alpha_signs, beta_signs))
    return tuple(result)


class ScalarEvolution:
    def __init__(self, simulator):
        self.simulator = simulator
        self._cache = {}
        self.excitations = None
        self.factors = ()
        self.native = None

    def plan(self, excitations):
        if (self.excitations is not None and len(self.excitations) == len(excitations)
                and all(a is b for a, b in zip(self.excitations, excitations))):
            return
        missing = {}
        for excitation in excitations:
            key = id(excitation)
            cached = self._cache.get(key)
            if cached is None or cached[0] is not excitation:
                orbitals = tuple(sorted({mode//2 for term in excitation.terms
                                         for mode in term.holes+term.particles}))
                missing.setdefault(orbitals, []).append(excitation)
        # Build maps together for all couplings on one orbital support, then
        # release the construction tables. The circuit factor order is kept.
        for group in missing.values():
            geometries = {}
            for excitation in group:
                self._cache[id(excitation)] = excitation, excitation_plan(self.simulator, excitation, geometries)
        self.excitations = tuple(excitations)
        self.factors = tuple(self._cache[id(e)][1] for e in excitations)
        self._cache = {id(e): self._cache[id(e)] for e in excitations}
        self.native = None
        module = self.simulator._native_module
        if module is not None and getattr(module, 'scalar_api_version', 0) == 2:
            definitions = []
            for factor in self.factors:
                spectra, blocks, ids = [], [], {}
                for block in factor:
                    spectrum = block.spectrum
                    if id(spectrum) not in ids:
                        ids[id(spectrum)] = len(spectra)
                        spectra.append((spectrum.generator, spectrum.frequencies, spectrum.basis))
                    blocks.append((ids[id(spectrum)], block.alpha, block.beta, block.alpha_signs, block.beta_signs))
                definitions.append((spectra, blocks))
            self.native = module.ScalarPlan(self.simulator.dimension, self.simulator.shape[1], definitions)

    @property
    def nbytes(self):
        arrays = {id(a): a for factor in self.factors for block in factor
                  for a in (block.alpha, block.beta, block.alpha_signs, block.beta_signs,
                            block.spectrum.generator, block.spectrum.frequencies,
                            block.spectrum.basis)}
        return sum(a.nbytes for a in arrays.values())

    def apply(self, state, parameters, excitations, *, adjoint=False, costate=None):
        self.plan(excitations)
        reverse = costate is not None
        gradient = np.zeros(len(parameters)) if reverse else None
        if (self.native is not None and state.flags.c_contiguous
                and state.dtype in (np.dtype('float64'), np.dtype('complex128'))
                and (not reverse or (costate.flags.c_contiguous
                     and costate.dtype in (np.dtype('float64'), np.dtype('complex128'))))):
            parameters = np.ascontiguousarray(parameters, dtype=float)
            threads = self.simulator.effective_rotation_threads()
            if reverse:
                self.native.reverse(parameters, state, costate, gradient, threads)
                return gradient
            self.native.forward(parameters, state, threads, adjoint=adjoint)
            return state
        order = reversed(range(len(parameters))) if adjoint or reverse else range(len(parameters))
        for i in order:
            theta = parameters[i]
            if theta == 0 and not reverse:
                continue
            matrices = {}
            for block in self.factors[i]:
                spectrum = block.spectrum
                indices, signs = block.addresses(self.simulator.shape[1]), block.signs()
                local = signs*state[indices]
                if reverse:
                    dual = signs*costate[indices]
                    gradient[i] += 2*np.vdot(dual, local @ spectrum.generator.T).real
                if theta == 0:
                    continue
                if id(spectrum) not in matrices:
                    matrices[id(spectrum)] = spectrum.matrix(-theta if adjoint or reverse else theta)
                matrix = matrices[id(spectrum)]
                state[indices] = signs*(local @ matrix.T)
                if reverse:
                    costate[indices] = signs*(dual @ matrix.T)
        return gradient if reverse else state
