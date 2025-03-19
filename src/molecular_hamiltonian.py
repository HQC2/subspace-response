#!/usr/bin/env python

import pyscf
import pennylane as qml
import numpy as np
import openfermion.ops.representations as reps
from openfermion.chem.molecular_data import spinorb_from_spatial
import openfermion.transforms

def get_molecular_hamiltonian(ucc):
    mf = ucc.mf
    mult = 1

    ncore = ucc.inactive_orbitals
    ncas = ucc.active_orbitals
    nelcas = ucc.active_electrons
    cas = pyscf.mcscf.CASCI(mf, nelecas=nelcas, ncas=ncas)
    one_body_integrals, constant = cas.get_h1cas()
    two_body_integrals = pyscf.ao2mo.full(ucc.m, ucc.mf.mo_coeff[:, ncore:ncore+ncas], aosym='1').reshape(ncas, ncas, ncas, ncas).transpose(0, 2, 3, 1)

    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
        one_body_integrals, two_body_integrals)
    terms_molecular_hamiltonian = reps.InteractionOperator(
        constant, one_body_coefficients, 1 / 2 * two_body_coefficients)
    fermionic_hamiltonian = openfermion.transforms.get_fermion_operator(terms_molecular_hamiltonian)
    H = openfermion.transforms.jordan_wigner(fermionic_hamiltonian)
    return qml.qchem.convert.import_operator(H), 2*ucc.active_orbitals

def get_PE_hamiltonian(ucc, v_PE):
    mf = ucc.mf
    mult = 1
    active_electrons = ucc.active_electrons
    active_orbitals = ucc.active_orbitals
        
    C = mf.mo_coeff
    h1e_mo = np.einsum('uj,uv,vi->ij', C, v_PE, C)
    
    occupied_indices, active_indices = qml.qchem.active_space(sum(mf.mol.nelec), mf.mol.nao, mult, active_electrons, active_orbitals)
    core_adjustment, one_body_integrals, two_body_integrals = reps.get_active_space_integrals(h1e_mo, np.zeros((ucc.m.nao,)*4), occupied_indices, active_indices)
    constant = core_adjustment # normally + mf.energy_nuc(), but that is already counted in the gas-phase hamiltonian
    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
        one_body_integrals, two_body_integrals)
    terms_molecular_hamiltonian = reps.InteractionOperator(
        constant, one_body_coefficients, 1 / 2 * two_body_coefficients)
    fermionic_hamiltonian = openfermion.transforms.get_fermion_operator(terms_molecular_hamiltonian)
    H = openfermion.transforms.jordan_wigner(fermionic_hamiltonian)
    return qml.qchem.convert.import_operator(H), 2*active_orbitals
