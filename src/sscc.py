#!/usr/bin/env python3

import uccsd
import solvers
import numpy as np
from pyscf.data.gyro import get_nuc_g_factor
from pyscf.data import nist

#-----------------------------------------------------------------------

def dso_integral(mol, orig1, orig2):
    '''Integral of vec{r}vec{r}/(|r-orig1|^3 |r-orig2|^3)
    Ref. JCP, 73, 5718'''
    NUMINT_GRIDS = 30
    from pyscf import gto
    t, w = np.polynomial.legendre.leggauss(NUMINT_GRIDS)
    a = (1+t)/(1-t) * .8
    w *= 2/(1-t)**2 * .8
    fakemol = gto.Mole()
    fakemol._atm = np.asarray([[0, 0, 0, 0, 0, 0]], dtype=np.int32)
    fakemol._bas = np.asarray([[0, 1, NUMINT_GRIDS, 1, 0, 3, 3+NUMINT_GRIDS, 0]],
                                 dtype=np.int32)
    p_cart2sph_factor = 0.488602511902919921
    fakemol._env = np.hstack((orig2, a**2, a**2*w*4/np.pi**.5/p_cart2sph_factor))
    fakemol._built = True

    pmol = mol + fakemol
    pmol.set_rinv_origin(orig1)
    # <nabla i, j | k>  k is a fictitious basis for numerical integraion
    mat1 = pmol.intor(mol._add_suffix('int3c1e_iprinv'), comp=3,
                      shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, pmol.nbas))
    # <i, j | nabla k>
    mat  = pmol.intor(mol._add_suffix('int3c1e_iprinv'), comp=3,
                      shls_slice=(mol.nbas, pmol.nbas, 0, mol.nbas, 0, mol.nbas))
    mat += mat1.transpose(0,3,1,2) + mat1.transpose(0,3,2,1)
    print(mat.shape)
    return mat

def _atom_gyro_list(mol):
    gyro = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb in mol.nucprop:
            prop = mol.nucprop[symb]
            mass = prop.get('mass', None)
            gyro.append(get_nuc_g_factor(symb, mass))
        else:
            # Get default isotope
            gyro.append(get_nuc_g_factor(symb))
    return np.array(gyro)

#-----------------------------------------------------------------------

ucc = uccsd.uccsd()
ucc.ground_state()

nuc_pair = [[0,1]]
# SSCC - DSO (expectation value)
ssc_dia = []
print('DSO:')
for (i,j) in nuc_pair:
    dso_ao = dso_integral(ucc.m, ucc.m.atom_coord(i), ucc.m.atom_coord(j)).reshape(9, *ucc.mf.mo_coeff.shape)
    a11 = -ucc.expectation_value(dso_ao).reshape(3,3)
    a11 = a11 - a11.trace() * np.eye(3)
    ssc_dia.append(a11)
#e11 = np.array(ssc_dia)*nist.ALPHA**4
e11 = np.array(ssc_dia)*nist.ALPHA**4*0

# SSCC - PSO (response)
h1 = []
d1 = []
print('PSO:')
for ia in range(ucc.m.natm):
    ucc.m.set_rinv_origin(ucc.m.atom_coord(ia))

    h1ao = -ucc.m.intor_asymmetric('int1e_prinvxp', 3)
    print('\nAO integral:\n', h1ao)
    property_gradient = ucc.property_gradient(h1ao)#, imag=True)
    print('\nProperty gradient:\n', property_gradient)
    h1.append(property_gradient)
    resp = []
    for pg in property_gradient:
        resp.append(solvers.cg(ucc.hvp, pg, verbose=True))
    d1.append(resp)
    #print('\nResponse ', (ia + 1), ': \n', resp)


for k, (i,j) in enumerate(nuc_pair):
    e11[k] = np.einsum('xi,yi->xy', d1[i], h1[j])

# unit conversions
nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
au2Hz = nist.HARTREE2J / nist.PLANCK
unit = au2Hz * nuc_magneton ** 2
iso_ssc = unit * np.einsum('kii->k', e11) / 3
natm = ucc.m.natm
ktensor = np.zeros((natm,natm))
for k, (i, j) in enumerate(nuc_pair):
    ktensor[i,j] = ktensor[j,i] = iso_ssc[k]
gyro = _atom_gyro_list(ucc.m)
jtensor = np.einsum('ij,i,j->ij', ktensor, gyro, gyro)
print('J tensor', jtensor)
