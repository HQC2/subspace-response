"""Streaming external equation residual norms for individual CSFs."""
import numpy as np


class ExternalResiduals:
    def __init__(self, candidates, selected):
        self.space = candidates.space
        self.external = np.ones(len(self.space), dtype=bool)
        self.external[np.asarray(selected,dtype=int)] = False
        self.squared = np.zeros(len(self.space))
        self.norms = []

    def add(self, images, sources=None):
        """Accumulate individual-CSF norms over roots or RHS branches.

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
        for indices,values in self.space.project_blocks(images):
            if not np.any(self.external[indices]):
                continue
            if sources is not None:
                values = sources[indices]-values
            mask = self.external[indices]
            values = values[mask]
            squares = abs(values)**2
            self.squared[indices[mask]] += np.sum(squares, axis=1)
            norms += np.sum(squares, axis=0)
        self.norms.extend(np.sqrt(norms).tolist())

    def passed(self, threshold):
        return np.flatnonzero(self.external & (self.squared >= threshold**2)).tolist()

    @property
    def maximum(self):
        return float(np.sqrt(np.max(self.squared,initial=0.)))
