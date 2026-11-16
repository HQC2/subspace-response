"""Occupation-block CSF transforms with byte-bounded numeric caches.

The canonical spin coupling is unchanged. No determinant-by-CSF matrix or
whole-pool Excitation/ExcitationTerm expansion is retained. Templates are shared
between occupation-equivalent blocks; determinant addresses are local to a block.
"""
from collections import OrderedDict
from collections.abc import Sequence
from itertools import combinations_with_replacement
from math import comb
from hashlib import sha256
import json
import numpy as np

from excitations import _singlet_coupling_template, ExcitationTerm


class _Excitations(Sequence):
    def __init__(self, space, ids=False):
        self.space, self.ids = space, ids

    def __len__(self):
        return len(self.space)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        canonical = self.space.indices[index]
        catalog = self.space.catalog
        block = int(np.searchsorted(catalog.starts, canonical, side='right')-1)
        holes, particles = catalog.blocks[block]
        coupling = int(canonical-catalog.starts[block])
        descriptor = _Descriptor(holes, particles, coupling, catalog, block)
        return descriptor.id if self.ids else descriptor


class _Descriptor:
    def __init__(self, holes, particles, coupling, catalog, block):
        self.spatial_holes, self.spatial_particles, self.coupling = holes, particles, coupling
        self.catalog, self.block_index = catalog, block

    @property
    def rank(self):
        return len(self.spatial_holes)

    @property
    def block(self):
        return self.spatial_holes, self.spatial_particles

    def _term_dicts(self):
        Q,hs,ps,order = self.catalog._template(self.block_index)
        orbitals = tuple(sorted(set(self.spatial_holes)))+tuple(sorted(set(self.spatial_particles)))
        for row in order[:,self.coupling]:
            if row < 0:
                break
            yield dict(holes=[2*orbitals[h//2]+int(h%2) for h in hs[row]],
                       particles=[2*orbitals[p//2]+int(p%2) for p in ps[row]],
                       coefficient=float(Q[row,self.coupling]))

    @property
    def terms(self):
        return tuple(ExcitationTerm(**term) for term in self._term_dicts())

    def to_dict(self):
        return dict(spatial_holes=list(self.spatial_holes),spatial_particles=list(self.spatial_particles),
                    coupling=self.coupling,terms=list(self._term_dicts()))

    @property
    def id(self):
        return sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()[:24]


class CompactResponseSpace:
    def __init__(self, simulator, max_rank, *, cache_bytes=32*1024**2):
        self.simulator = simulator
        self.hf_index = int(simulator.indices[np.flatnonzero(simulator.initial)[0]])
        self.catalog = self
        self.blocks, starts = [], [0]
        occupied, orbitals = simulator.integrals.electrons//2, simulator.n_orbitals
        maximum = min(max_rank, simulator.integrals.electrons,
                      simulator.n_modes-simulator.integrals.electrons)
        for rank in range(1, maximum+1):
            holes = [h for h in combinations_with_replacement(range(occupied), rank)
                     if all(h.count(i) <= 2 for i in h)]
            particles = [p for p in combinations_with_replacement(range(occupied, orbitals), rank)
                         if all(p.count(i) <= 2 for i in p)]
            for h in holes:
                for p in particles:
                    opens = sum(x.count(i) == 1 for x in (h,p) for i in set(x))
                    count = comb(opens, opens//2)-comb(opens, opens//2+1)
                    self.blocks.append((h,p))
                    starts.append(starts[-1]+count)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.indices = np.arange(starts[-1])
        self._templates, self._layouts = OrderedDict(), OrderedDict()
        self.cache_bytes = int(cache_bytes)
        self._cache_sizes = {}
        self.excitations, self.ids = _Excitations(self), _Excitations(self, ids=True)
        self._groups = None

    def __len__(self):
        return len(self.indices)

    def restrict(self, indices):
        from copy import copy
        result = copy(self)
        result.indices = self.indices[np.asarray(indices, dtype=int)]
        result.excitations, result.ids = _Excitations(result), _Excitations(result, ids=True)
        result._groups = None
        return result

    def groups(self):
        if self._groups is None:
            grouped = {}
            for local, canonical in enumerate(self.indices):
                block = int(np.searchsorted(self.catalog.starts, canonical, side='right')-1)
                grouped.setdefault(block, []).append(local)
            self._groups = [(block, np.asarray(local, dtype=int)) for block,local in grouped.items()]
        return self._groups

    def _cache(self, cache, key, value):
        identity = id(cache)
        size = self.catalog._cache_sizes.get(identity,0)
        if key in cache:
            size -= sum(a.nbytes for a in cache[key])
        cache[key] = value
        size += sum(a.nbytes for a in value)
        cache.move_to_end(key)
        while size > self.catalog.cache_bytes//2:
            _, entry = cache.popitem(last=False)
            size -= sum(a.nbytes for a in entry)
        self.catalog._cache_sizes[identity] = size
        return value

    def _template(self, block):
        catalog = self.catalog
        holes, particles = catalog.blocks[block]
        orbitals = tuple(sorted(set(holes)))+tuple(sorted(set(particles)))
        labels = {o:i for i,o in enumerate(orbitals)}
        key = tuple(labels[h] for h in holes), tuple(labels[p] for p in particles)
        if key in catalog._templates:
            catalog._templates.move_to_end(key)
            return catalog._templates[key]
        else:
            opens = sum(x.count(i) == 1 for x in (holes,particles) for i in set(x))
            # Template construction itself must fit, even before it is cached.
            spin_dimension = comb(opens, opens//2)
            singlets = int(catalog.starts[block+1]-catalog.starts[block])
            if 128*spin_dimension*singlets > catalog.cache_bytes:
                raise ValueError('Spin-coupling template exceeds the LR transform budget; increase --lr-memory-mb')
            template = _singlet_coupling_template.__wrapped__(*key)
            monomials = sorted({m for vector in template for m,c in vector})
            address = {m:i for i,m in enumerate(monomials)}
            Q = np.zeros((len(monomials),len(template)))
            order = np.full(Q.shape,-1,dtype=np.int32)
            for j,vector in enumerate(template):
                for i,(monomial,coefficient) in enumerate(vector):
                    Q[address[monomial],j] = coefficient
                    order[i,j] = address[monomial]
            hs = np.asarray([m[0] for m in monomials],dtype=int)
            ps = np.asarray([m[1] for m in monomials],dtype=int)
            return self._cache(catalog._templates,key,(Q,hs,ps,order))

    def _transform(self, block):
        catalog,sim = self.catalog,self.simulator
        holes,particles = catalog.blocks[block]
        orbitals = tuple(sorted(set(holes)))+tuple(sorted(set(particles)))
        Q,hs,ps,_ = self._template(block)
        if block in catalog._layouts:
            catalog._layouts.move_to_end(block)
            rows, phases = catalog._layouts[block]
        else:
            modes = np.asarray([2*o+s for o in orbitals for s in (0,1)])
            states = np.full(len(hs), catalog.hf_index,dtype=np.uint64)
            phases = np.ones(len(hs))
            # apply_fermion_word acts from right to left: ascending holes,
            # followed by descending particles for a canonical monomial.
            for local in list(hs.T)+list(ps[:, ::-1].T):
                shifts = np.asarray(sim.n_modes-modes[local],dtype=np.uint64)
                phases *= 1-2*(np.bitwise_count(states >> shifts).astype(int)%2)
                states ^= np.left_shift(np.uint64(1),shifts-1)
            positions = np.searchsorted(sim._sorted_indices, states.astype(np.int64))
            rows = sim._index_order[positions]
            self._cache(catalog._layouts,block,(rows,phases))
        return Q, rows, phases

    def expand(self, coefficients):
        coefficients = np.asarray(coefficients)
        single = coefficients.ndim == 1
        if single:
            coefficients = coefficients[:,None]
        result = np.zeros((self.simulator.dimension,coefficients.shape[1]),dtype=np.result_type(coefficients,float))
        for block, local in self.groups():
            if not np.any(coefficients[local]):
                continue
            Q, rows, phases = self._transform(block)
            columns = self.indices[local]-self.catalog.starts[block]
            result[rows] = phases[:,None]*(Q[:,columns] @ coefficients[local])
        return result[:,0] if single else result

    def project_blocks(self, states):
        states = np.asarray(states)
        for block, local in self.groups():
            Q, rows, phases = self._transform(block)
            columns = self.indices[local]-self.catalog.starts[block]
            weighted = phases*states[rows] if states.ndim == 1 else phases[:,None]*states[rows]
            yield local, Q[:,columns].T @ weighted

    def project(self, states):
        states = np.asarray(states)
        result = np.empty((len(self),)+states.shape[1:],dtype=np.result_type(states,float))
        for local, values in self.project_blocks(states):
            result[local] = values
        return result
