"""Streaming external equation residual norms for individual CSFs."""
import numpy as np


class ExternalResiduals:
    def __init__(self, candidates, selected):
        self.space = candidates.space
        self.external = np.ones(len(self.space), dtype=bool)
        self.external[np.asarray(selected,dtype=int)] = False
        self.squared = np.zeros(len(self.space))
        self.norms = []
        self.norms_are_bounds = False
        self._omitted_maximum = 0.

    def add(self, images, sources=None, *, threshold=None):
        """Accumulate individual-CSF norms over roots or RHS branches.

        Only a block-sized projection is live; never an M by P coupling table.
        With no source this is the external Ritz residual (sign immaterial).
        Frequency/eigenvalue identity terms vanish outside the selected space.
        ``threshold`` permits discarding certified subthreshold blocks in one
        compact-space batch containing all RHS images. Scores in those blocks
        are omitted; ``norms`` and ``maximum`` then report conservative bounds,
        indicated by ``norms_are_bounds``. Query ``passed`` at the same threshold.
        """
        images = np.asarray(images)
        if images.ndim == 1:
            images = images[:,None]
        if sources is not None:
            sources = np.asarray(sources)
            if sources.ndim == 1:
                sources = sources[:,None]
        norms = np.zeros(images.shape[1])
        if not np.any(self.external):
            self.norms.extend(norms.tolist())
            return
        space = self.space
        omitted = np.zeros(images.shape[1])
        if threshold is not None:
            # Use only with one batch containing every property/frequency image.
            # Triangle/Frobenius bounds certify that an entire block cannot
            # contain a passing individual CSF before constructing its template.
            catalog = space.catalog
            bounds = space.occupation_norms(images)
            blocks = np.searchsorted(catalog.starts,space.indices,side='right')-1
            if sources is not None:
                bounds += np.column_stack([np.sqrt(np.bincount(blocks,weights=abs(v)**2,
                    minlength=len(catalog.blocks))) for v in sources.T])
            squared_bounds = np.sum(bounds**2,axis=1)
            active = squared_bounds >= threshold**2
            external_blocks = np.bincount(blocks,weights=self.external,minlength=len(active)) > 0
            skipped = ~active & external_blocks
            omitted = np.sum(bounds[skipped]**2,axis=0)
            self._omitted_maximum = float(np.sqrt(np.max(squared_bounds[skipped],initial=0.)))
            self.norms_are_bounds = bool(np.any(skipped))
            selected = np.flatnonzero(active[blocks] & self.external)
            space = space.restrict(selected)
        for indices,values in space.project_blocks(images):
            if threshold is not None:
                indices = selected[indices]
            if not np.any(self.external[indices]):
                continue
            if sources is not None:
                values = sources[indices]-values
            mask = self.external[indices]
            values = values[mask]
            squares = abs(values)**2
            self.squared[indices[mask]] += np.sum(squares, axis=1)
            norms += np.sum(squares, axis=0)
        self.norms.extend(np.sqrt(norms+omitted).tolist())

    def passed(self, threshold):
        return np.flatnonzero(self.external & (self.squared >= threshold**2)).tolist()

    @property
    def maximum(self):
        return max(self._omitted_maximum,float(np.sqrt(np.max(self.squared,initial=0.))))
