"""Streaming external equation residual norms for complete spatial blocks."""
import numpy as np


class ExternalResiduals:
    def __init__(self, candidates, selected):
        self.space = candidates.space
        self.external = np.ones(len(self.space), dtype=bool)
        self.external[np.asarray(selected,dtype=int)] = False
        self.groups = self.space.groups()
        self.squared = np.zeros(len(self.groups))
        self.norms = []

    def add(self, images, sources=None):
        """Accumulate block Frobenius norms over column roots or RHS branches.

        Only a block-sized projection is live; never an M by P coupling table.
        With no source this is the external Ritz residual (sign immaterial).
        Frequency/eigenvalue identity terms vanish outside the selected space.
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
        for block,(indices,values) in enumerate(self.space.project_blocks(images)):
            if not np.any(self.external[indices]):
                continue
            if sources is not None:
                values = sources[indices]-values
            values = values[self.external[indices]]
            squares = np.sum(abs(values)**2,axis=0)
            self.squared[block] += np.sum(squares)
            norms += squares
        self.norms.extend(np.sqrt(norms).tolist())

    def passed(self, threshold):
        chosen = []
        for score,(_,indices) in zip(self.squared,self.groups):
            if score >= threshold**2:
                chosen.extend(indices[self.external[indices]])
        return chosen

    @property
    def maximum(self):
        return float(np.sqrt(np.max(self.squared,initial=0.)))
