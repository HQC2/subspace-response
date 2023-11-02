#!/usr/bin/env python

import uccsd
import solvers
import pennylane as qml
import numpy as np
import numpy
import pyscf
from pyscf.data.gyro import get_nuc_g_factor
from pyscf.data import nist


#symbols = ['O', 'H', 'H']
#xyz = qml.numpy.array([
#[0.0000000000,   -0.0000000000,    0.0664432016],
#[0.0000000000,    0.7532904501,   -0.5271630249],
#[0.0000000000,   -0.7532904501,   -0.5271630249]
#                ], requires_grad=False) * 1.8897259886

symbols = ['H', 'H', 'H', 'H']
geometry = qml.numpy.array([
[0.0000000000,   -0.0000000000,    0.0],
[0.0000000000,   -0.0000000000,    2.0],
[1.5000000000,   -0.0000000000,    0.0],
[1.5000000000,   -0.0000000000,    2.0],
                ], requires_grad=False) 

charge = 0
basis = 'STO-3G'

def dso_integral(mol, orig1, orig2):
    '''Integral of vec{r}vec{r}/(|r-orig1|^3 |r-orig2|^3)
    Ref. JCP, 73, 5718'''
    NUMINT_GRIDS = 30
    from pyscf import gto
    t, w = numpy.polynomial.legendre.leggauss(NUMINT_GRIDS)
    a = (1+t)/(1-t) * .8
    w *= 2/(1-t)**2 * .8
    fakemol = gto.Mole()
    fakemol._atm = numpy.asarray([[0, 0, 0, 0, 0, 0]], dtype=numpy.int32)
    fakemol._bas = numpy.asarray([[0, 1, NUMINT_GRIDS, 1, 0, 3, 3+NUMINT_GRIDS, 0]],
                                 dtype=numpy.int32)
    p_cart2sph_factor = 0.488602511902919921
    fakemol._env = numpy.hstack((orig2, a**2, a**2*w*4/numpy.pi**.5/p_cart2sph_factor))
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
    return numpy.array(gyro)

def convert_unit(e11):
    # unit conversions
    e11 = e11*nist.ALPHA**4
    nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
    au2Hz = nist.HARTREE2J / nist.PLANCK
    unit = au2Hz * nuc_magneton ** 2
    iso_ssc = unit * numpy.einsum('kii->k', e11) / 3
    natm = ucc.m.natm
    ktensor = numpy.zeros((natm,natm))
    for k, (i, j) in enumerate(nuc_pair):
        ktensor[i,j] = ktensor[j,i] = iso_ssc[k]
    gyro = _atom_gyro_list(ucc.m)
    jtensor = numpy.einsum('ij,i,j->ij', ktensor, gyro, gyro)
    return jtensor

ucc = uccsd.uccsd(symbols, geometry, charge, basis)
theta = np.array([-1.35066821e-16, -4.54632328e-17, -4.86482370e-17, -1.42047871e-16,
  1.03048404e-01,  2.26491572e-16,  1.13066447e-16, -1.87589773e-01,
 -5.11526101e-02,  4.97979006e-02,  3.49274247e-16,  2.14070553e-01,
  1.15892459e-16,  9.26284955e-02,])
ucc.theta = theta
ucc.ground_state()

nuc_pair = [(i,j) for i in range(ucc.m.natm) for j in range(i)]
print(nuc_pair)
# SSCC - DSO (expectation value)
ssc_dia = []
for (i,j) in nuc_pair:
    dso_ao = dso_integral(ucc.m, ucc.m.atom_coord(i), ucc.m.atom_coord(j)).reshape(9, *ucc.mf.mo_coeff.shape)
    a11 = -ucc.expectation_value(dso_ao).reshape(3,3)
    a11 = a11 - a11.trace() * np.eye(3)
    ssc_dia.append(a11)
e11_dso = np.array(ssc_dia)

# SSCC - PSO (response)
h1 = []
d1 = []
for ia in range(ucc.m.natm):
    ucc.m.set_rinv_origin(ucc.m.atom_coord(ia))

    h1ao = ucc.m.intor_asymmetric('int1e_prinvxp', 3)
    print('AO integral', h1ao)
    property_gradient = ucc.property_gradient(h1ao, approach='statevector')
    print('Property gradient', property_gradient)
    h1.append(property_gradient)
    d = []
    for pg in property_gradient:
        d.append(solvers.cg(ucc.hvp, pg, verbose=True))
    d1.append(d)

e11_pso = np.zeros_like(e11_dso)
for k, (i,j) in enumerate(nuc_pair):
    e11_pso[k] = -np.einsum('xi,yi->xy', d1[i], h1[j]) # minus because imag

# SSCC - FC + SD (response)
#...

j_tensor_dso = convert_unit(e11_dso)
j_tensor_pso = convert_unit(e11_pso)
#j_tensor_fcsd = convert_unit(e11_fcsd)
j_tensor_total = j_tensor_dso + j_tensor_pso # + j_tensor_fcsd

print('SSCC (in Hz):')
for (i,j) in nuc_pair:
    print(f'{i} {j}: DSO={j_tensor_dso[i,j]:.3f}, PSO={j_tensor_pso[i,j]:.3f}, Total={j_tensor_total[i,j]:.3f}')
