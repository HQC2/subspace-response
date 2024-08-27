#!/usr/bin/env python

import pyscf
import pennylane as qml
import numpy as np
import openfermion.ops.representations as reps
from openfermion.chem.molecular_data import spinorb_from_spatial
import openfermion.transforms

def get_molecular_hamiltonian(ucc, active_electrons, active_orbitals):
    mf = ucc.mf
    mult = 1
        
    C = mf.mo_coeff
    h1e_ao = mf.get_hcore()
    h2e_ao = mf.mol.intor('int2e')
    h1e_mo = np.einsum('uj,uv,vi->ij', C, h1e_ao, C)
    h2e_mo = np.einsum('ai,bj,ck,dl,abcd->ijkl', C, C, C, C, h2e_ao, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    ucc.h1e_mo = h1e_mo
    ucc.h2e_mo = h2e_mo

    occupied_indices, active_indices = qml.qchem.active_space(sum(mf.mol.nelec), mf.mol.nao, mult, active_electrons, active_orbitals)
    core_adjustment, one_body_integrals, two_body_integrals = reps.get_active_space_integrals(h1e_mo, h2e_mo.transpose(0, 2, 3, 1), occupied_indices, active_indices)
    constant = mf.energy_nuc() + core_adjustment
    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
        one_body_integrals, two_body_integrals)
    terms_molecular_hamiltonian = reps.InteractionOperator(
        constant, one_body_coefficients, 1 / 2 * two_body_coefficients)
    fermionic_hamiltonian = openfermion.transforms.get_fermion_operator(terms_molecular_hamiltonian)
    H = openfermion.transforms.jordan_wigner(fermionic_hamiltonian)
    return qml.qchem.convert.import_operator(H), 2*active_orbitals

def get_PE_hamiltonian(ucc, active_electrons, active_orbitals, v_PE):
    mf = ucc.mf
    mult = 1
        
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
