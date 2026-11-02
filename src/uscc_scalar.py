"""Exact scalar factors on small active-mode blocks with compressed spectators.

Move an excitation's active fermion modes ahead of the spectators. In that
ordering its generator is G_active tensor I. The permutation's fermionic
parity is retained in the embedding, so no Jordan-Wigner strings are dropped.
Only connected, nonzero blocks of G_active are diagonalized, once per coupling
template. A plan stores short alpha/beta string maps rather than sector-sized
matrices or one copy of each determinant address per spectator pair.
"""
from functools import lru_cache

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
    """Real spectral projectors: exp(tG)=I+sum (cos-1)P+sin K."""
    def __init__(self, generator):
        self.generator = np.ascontiguousarray(generator)
        size = len(generator)
        if size == 2:
            self.frequencies = np.array([generator[0, 1]])
            self.cosine = np.eye(2)[None, :, :]
            self.sine = np.array([[[0., 1.], [-1., 0.]]])
        else:
            t, q = schur(generator, output='real')
            frequencies, cosine, sine = [], [], []
            i = 0
            while i < size-1:
                if abs(t[i, i+1]) < 1e-13:
                    i += 1
                    continue
                a, b = q[:, i], q[:, i+1]
                frequencies.append(t[i, i+1])
                cosine.append(np.outer(a, a)+np.outer(b, b))
                sine.append(np.outer(a, b)-np.outer(b, a))
                i += 2
            self.frequencies = np.array(frequencies)
            self.cosine = np.array(cosine).reshape(-1, size, size)
            self.sine = np.array(sine).reshape(-1, size, size)
        for a in (self.generator, self.frequencies, self.cosine, self.sine):
            a.flags.writeable = False

    def matrix(self, theta):
        angles = theta*self.frequencies
        # -2 sin²(x/2) retains accuracy of cos(x)-1 near zero.
        return (np.eye(len(self.generator))
                + np.einsum('k,kij->ij', -2*np.sin(angles/2)**2, self.cosine)
                + np.einsum('k,kij->ij', np.sin(angles), self.sine))


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
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        members = order[start:stop]
        if len(members) < 2:
            continue
        matrix = generator[np.ix_(members, members)].toarray()
        key = matrix.shape, matrix.tobytes()
        if key not in spectra:
            spectra[key] = Spectrum(matrix)
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
        alpha_table, alpha_phase = geometry.tables(remaining_a)
        beta_table, beta_phase = geometry.tables(remaining_b)
        alpha = np.ascontiguousarray(alpha_table[alpha_codes].transpose(0, 2, 1))
        beta = np.ascontiguousarray(beta_table[beta_codes].transpose(0, 2, 1))
        alpha_signs = np.ascontiguousarray(alpha_phase[parity_codes].transpose(0, 2, 1))
        beta_signs = np.ascontiguousarray(beta_phase[parity_codes].transpose(0, 2, 1))
        result.append(Embedding(spectrum, alpha, beta, alpha_signs, beta_signs))
    return tuple(result)


class ScalarEvolution:
    def __init__(self, simulator):
        self.simulator = simulator
        self._cache = {}
        self._geometries = {}
        self.excitations = None
        self.factors = ()
        self.native = None

    def plan(self, excitations):
        if (self.excitations is not None and len(self.excitations) == len(excitations)
                and all(a is b for a, b in zip(self.excitations, excitations))):
            return
        factors = []
        for excitation in excitations:
            key = id(excitation)
            cached = self._cache.get(key)
            if cached is None or cached[0] is not excitation:
                cached = self._cache[key] = excitation, excitation_plan(self.simulator, excitation, self._geometries)
            factors.append(cached[1])
        self.excitations, self.factors = tuple(excitations), tuple(factors)
        self._cache = {id(e): self._cache[id(e)] for e in excitations}
        self.native = None
        module = self.simulator._native_module
        if module is not None and getattr(module, 'scalar_api_version', 0) == 1:
            definitions = []
            for factor in self.factors:
                spectra, blocks, ids = [], [], {}
                for block in factor:
                    spectrum = block.spectrum
                    if id(spectrum) not in ids:
                        ids[id(spectrum)] = len(spectra)
                        spectra.append((spectrum.generator, spectrum.frequencies, spectrum.cosine, spectrum.sine))
                    blocks.append((ids[id(spectrum)], block.alpha, block.beta, block.alpha_signs, block.beta_signs))
                definitions.append((spectra, blocks))
            self.native = module.ScalarPlan(self.simulator.dimension, self.simulator.shape[1], definitions)

    @property
    def nbytes(self):
        arrays = {id(a): a for factor in self.factors for block in factor
                  for a in (block.alpha, block.beta, block.alpha_signs, block.beta_signs,
                            block.spectrum.generator, block.spectrum.frequencies,
                            block.spectrum.cosine, block.spectrum.sine)}
        for geometry in self._geometries.values():
            for a in (geometry.codes, geometry.counts, geometry.masks, geometry.gauges):
                arrays[id(a)] = a
            for columns, phases in geometry._tables.values():
                arrays[id(columns)], arrays[id(phases)] = columns, phases
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
