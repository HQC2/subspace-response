#!/usr/bin/env python

from dataclasses import dataclass
from itertools import combinations

import pyscf
import pennylane as qml
import numpy as np
import openfermion.ops.representations as reps
from openfermion.chem.molecular_data import spinorb_from_spatial
import openfermion.transforms
from excitations import ExcitationTerm


@dataclass(frozen=True)
class ActiveSpaceIntegrals:
    """Effective active Hamiltonian: h[p,q], g[p,q,r,s]=(p s|q r).

    spinorb_from_spatial supplies the tensor V in
    H = constant + h[p,q] a†p aq + (1/2) V[p,q,r,s] a†p a†q ar as.
    Screening uses h and the canonical antisymmetrized quartic coefficients,
    never the Fock matrix or a determinant coupling containing spectators.
    """
    one_body: np.ndarray
    two_body: np.ndarray
    constant: float
    electrons: int

    def __post_init__(self):
        if (np.iscomplexobj(self.one_body) and np.any(np.imag(self.one_body))) or (np.iscomplexobj(self.two_body) and np.any(np.imag(self.two_body))):
            raise ValueError('The initial USCC implementation requires real integrals')
        h = np.array(np.real(self.one_body), dtype=float, copy=True)
        g = np.array(np.real(self.two_body), dtype=float, copy=True)
        if h.ndim != 2 or h.shape[0] != h.shape[1] or not h.shape[0]:
            raise ValueError('one_body must be a nonempty square matrix')
        if g.shape != h.shape*2 or not np.all(np.isfinite(h)) or not np.all(np.isfinite(g)):
            raise ValueError('Invalid finite active-space integral dimensions')
        if not np.allclose(h, h.T) or not np.isfinite(self.constant):
            raise ValueError('Expected a real Hermitian active Hamiltonian')
        if not isinstance(self.electrons, (int, np.integer)) or self.electrons < 0 or self.electrons % 2 or self.electrons > 2*len(h):
            raise ValueError('Invalid closed-shell active electron count')
        h.flags.writeable = False
        g.flags.writeable = False
        object.__setattr__(self, 'one_body', h)
        object.__setattr__(self, 'two_body', g)
        object.__setattr__(self, 'electrons', int(self.electrons))
        object.__setattr__(self, 'constant', float(self.constant))

    @property
    def qubits(self):
        return 2*len(self.one_body)

    def spin_integrals(self):
        return spinorb_from_spatial(self.one_body, self.two_body)

    def excitation_gaps(self, excitations):
        """HF Fock excitation gaps in the active reference, without a floor."""
        eri = self.two_body.transpose(0, 3, 1, 2)
        energies = np.diag(self.one_body).copy()
        for i in range(self.electrons//2):
            energies += 2*np.diag(eri[:, :, i, i])-np.diag(eri[:, i, i, :])
        return np.array([sum(energies[p] for p in e.spatial_particles)
                         - sum(energies[h] for h in e.spatial_holes)
                         for e in excitations], dtype=float)

    def hamiltonian(self):
        h, v = self.spin_integrals()
        interaction = reps.InteractionOperator(self.constant, h, .5*v)
        fermion = openfermion.transforms.get_fermion_operator(interaction)
        return qml.qchem.convert.import_operator(openfermion.transforms.jordan_wigner(fermion))

    def excitation_channels(self):
        """Canonical occupied-to-virtual one-/two-body integral coefficients."""
        h, v = self.spin_integrals()
        occ = range(self.electrons)
        vir = range(self.electrons, self.qubits)
        channels = []
        for i in occ:
            for a in vir:
                if i % 2 == a % 2:
                    channels.append(ExcitationTerm((i,), (a,), h[a,i]))
        for i,j in combinations(occ, 2):
            for a,b in combinations(vir, 2):
                if i % 2 + j % 2 != a % 2 + b % 2:
                    continue
                coefficient = .5*(v[a,b,j,i]-v[b,a,j,i]-v[a,b,i,j]+v[b,a,i,j])
                channels.append(ExcitationTerm((i,j), (a,b), coefficient))
        return tuple(channels)


def get_active_space_integrals(ucc):
    ncore, ncas = ucc.inactive_orbitals, ucc.active_orbitals
    cas = pyscf.mcscf.CASCI(ucc.mf, nelecas=ucc.active_electrons, ncas=ncas)
    one_body, constant = cas.get_h1cas()
    coefficients = ucc.mf.mo_coeff[:, ncore:ncore+ncas]
    two_body = pyscf.ao2mo.full(ucc.m, coefficients, aosym='1').reshape((ncas,)*4).transpose(0,2,3,1)
    return ActiveSpaceIntegrals(one_body, two_body, constant, ucc.active_electrons)


def get_molecular_hamiltonian(ucc):
    integrals = get_active_space_integrals(ucc)
    return integrals.hamiltonian(), integrals.qubits
