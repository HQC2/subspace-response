#!/usr/bin/env python

# based on https://docs.pennylane.ai/en/stable/code/api/pennylane.UCCSD.html

import pennylane as qml
import numpy as np
from pennylane._grad import grad as get_gradient
from scipy.optimize import minimize
import periodictable
import functools
from uccsd_circuits import UCCSD, UCCSD_exc
import pyscf
import excitations
import time


def uccsd_ground_state(symbols, geometry, charge, basis=None, min_method=None):
    if basis is None:
        basis = 'STO-3G'

    H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge, method='pyscf', basis=basis)

    electrons = sum([periodictable.elements.__dict__[symbol].number for symbol in symbols]) - charge

    hf_state = qml.qchem.hf_state(electrons, qubits)
    singles, doubles, parameter_map = excitations.spin_adapted_excitations(electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    dev = qml.device("lightning.qubit", wires=qubits)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params, wires, s_wires, d_wires, parameter_map, hf_state):
        UCCSD(params, wires, s_wires, d_wires, parameter_map, hf_state)
        return qml.expval(H)

    def energy(params):
        params = qml.numpy.array(params)
        energy = circuit(params, range(qubits), s_wires, d_wires, parameter_map, hf_state)
        print('energy = ', energy)
        return energy

    def jac(params):
        params = qml.numpy.array(params)
        grad = get_gradient(circuit)(params, range(qubits), s_wires, d_wires, parameter_map, hf_state)[0]
        return grad
    
    num_params = max(parameter_map[len(parameter_map)-1][0]) + 1
    params = qml.numpy.zeros(num_params)
    t1 = time.time()
    
    print(num_params)


    if min_method is not None:
        res = minimize(energy, jac=jac, x0=params, method=min_method, tol=1e-12)
    else:
        res = minimize(energy, jac=jac, x0=params, method='slsqp', tol=1e-12)
    t2 = time.time()
    print('Optimizing ground state took', t2 - t1, 'seconds')

    theta_opt = res.x
    def spinstring(l):
        string = ''
        for i in l:
            string += 'ab'[i%2]
        return string
    for i, single in enumerate(singles):
        pmap = parameter_map[i][0]
        print(single, spinstring(single), pmap, theta_opt[pmap])
    for i, double in enumerate(doubles):
        pmap = parameter_map[i+len(singles)][0]
        print(double, spinstring(double), pmap, theta_opt[pmap])
    print(theta_opt)
    print(res)
    return theta_opt

def uccsd_hvp(symbols, geometry, charge, theta_opt, basis=None, h=1e-6):
    if basis is None:
        basis = 'STO-3G'
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge, method='pyscf', basis=basis)

    electrons = sum([periodictable.elements.__dict__[symbol].number for symbol in symbols]) - charge

    hf_state = qml.qchem.hf_state(electrons, qubits)
    singles, doubles, parameter_map = excitations.spin_adapted_excitations(electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    dev = qml.device("lightning.qubit", wires=qubits)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit_sd(params, wires, s_wires, d_wires, parameter_map, hf_state):
        UCCSD_exc(params, wires, s_wires, d_wires, parameter_map, hf_state)
        return qml.expval(H)

    def jac_sd(params_excitation):
        params_excitation = qml.numpy.array(params_excitation)
        grad = get_gradient(circuit_sd)(params_excitation, range(qubits), s_wires, d_wires, parameter_map, hf_state)[0]
        return grad

    def jac_exc(v, theta_opt=theta_opt, full_grad=jac_sd):
        # gradient on SingleExcitation + DoubleExcitation parameter part
        params = qml.numpy.zeros(len(theta_opt) + len(v))
        params[:len(theta_opt)] = theta_opt
        params[len(theta_opt):] = v
        fg = full_grad(params)
        grad = fg[len(theta_opt):]
        return grad

    def hessian_vector_product(v, grad, theta, h):
        hvp = np.zeros_like(v)
        if len(v.shape) == 1:
            hvp = (grad(theta+h*v) - grad(theta-h*v)) / (2*h)
        elif len(v.shape) == 2:
            for i in range(v.shape[1]):
                hvp[:, i] =  (grad(theta+h*v[:,i]) - grad(theta-h*v[:,i])) / (2*h)
        else:
            raise ValueError
        return hvp

    num_params = max(parameter_map[len(parameter_map)-1][0]) + 1
    dim = num_params
    hvp = functools.partial(hessian_vector_product, grad=jac_exc, theta=qml.numpy.zeros(dim), h=h)
    return hvp

def uccsd_spin_squared(symbols, geometry, charge, theta_opt, v, basis=None):
    if basis is None:
        basis = 'STO-3G'
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge, method='pyscf', basis=basis)

    electrons = sum([periodictable.elements.__dict__[symbol].number for symbol in symbols]) - charge

    hf_state = qml.qchem.hf_state(electrons, qubits)
    singles, doubles, parameter_map = excitations.spin_adapted_excitations(electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    params = qml.numpy.zeros(len(theta_opt) + len(v))
    params[:len(theta_opt)] = theta_opt
    params[len(theta_opt):] = v

    dev = qml.device("lightning.qubit", wires=qubits)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params, wires, s_wires, d_wires, parameter_map, hf_state, electrons, qubits):
        UCCSD_exc(params, wires, s_wires, d_wires, parameter_map, hf_state)
        return qml.expval(qml.qchem.spin2(electrons, qubits))
    @qml.qnode(dev, diff_method="adjoint")
    def circuitz(params, wires, s_wires, d_wires, parameter_map, hf_state, electrons, qubits):
        UCCSD_exc(params, wires, s_wires, d_wires, parameter_map, hf_state)
        return qml.expval(qml.qchem.spinz(qubits))
    
    return circuit(params, range(qubits), s_wires, d_wires, parameter_map, hf_state, electrons, qubits)

def uccsd_dipole_property_gradient(symbols, geometry, charge, theta_opt, basis=None):
    if basis is None:
        basis = 'STO-3G'
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge, method='pyscf', basis=basis)

    electrons = sum([periodictable.elements.__dict__[symbol].number for symbol in symbols]) - charge

    hf_state = qml.qchem.hf_state(electrons, qubits)
    singles, doubles, parameter_map = excitations.spin_adapted_excitations(electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    dev = qml.device("lightning.qubit", wires=qubits)
    atom_str = ''
    for symbol, coord in zip(symbols, geometry):
        atom_str += f'{symbol} {coord[0]} {coord[1]} {coord[2]}; '
        print(atom_str)
    m = pyscf.M(atom=atom_str, basis=basis, charge=charge, unit='bohr')
    mf = pyscf.scf.RHF(m)
    mf.kernel()
    mo_dipole_integrals = np.einsum('uj,xuv,vi->xij', mf.mo_coeff, m.intor('int1e_r'), mf.mo_coeff)
    try:
        dipole_x = qml.qchem.qubit_observable(qml.qchem.fermionic_observable(np.array([0.0]), mo_dipole_integrals[0]))
    except IndexError:
        dipole_x = None
        gradx = None
    try:
        dipole_y = qml.qchem.qubit_observable(qml.qchem.fermionic_observable(np.array([0.0]), mo_dipole_integrals[1]))
    except IndexError:
        dipole_y = None
        grady = None
    try:
        dipole_z = qml.qchem.qubit_observable(qml.qchem.fermionic_observable(np.array([0.0]), mo_dipole_integrals[2]))
    except IndexError:
        dipole_z = None
        gradz = None

    @qml.qnode(dev, diff_method="adjoint")
    def circuit_dipole_x(params, wires, s_wires, d_wires, parameter_map, hf_state):
        UCCSD_exc(params, wires, s_wires, d_wires, parameter_map, hf_state)
        return qml.expval(dipole_x)
    @qml.qnode(dev, diff_method="adjoint")
    def circuit_dipole_y(params, wires, s_wires, d_wires, parameter_map, hf_state):
        UCCSD_exc(params, wires, s_wires, d_wires, parameter_map, hf_state)
        return qml.expval(dipole_y)
    @qml.qnode(dev, diff_method="adjoint")
    def circuit_dipole_z(params, wires, s_wires, d_wires, parameter_map, hf_state):
        UCCSD_exc(params, wires, s_wires, d_wires, parameter_map, hf_state)
        return qml.expval(dipole_z)

    params_excitation = qml.numpy.zeros(len(theta_opt)*2)
    params_excitation[:len(theta_opt)] = theta_opt
    if dipole_x:
        print('X:', circuit_dipole_x(params_excitation, range(qubits), s_wires, d_wires, parameter_map, hf_state))
        gradx = get_gradient(circuit_dipole_x)(params_excitation, range(qubits), s_wires, d_wires, parameter_map, hf_state)[0]
        gradx = gradx[len(theta_opt):]
    if dipole_y:
        print('Y:', circuit_dipole_y(params_excitation, range(qubits), s_wires, d_wires, parameter_map, hf_state))
        grady = get_gradient(circuit_dipole_y)(params_excitation, range(qubits), s_wires, d_wires, parameter_map, hf_state)[0]
        grady = grady[len(theta_opt):]
    if dipole_z:
        print('Z:', circuit_dipole_z(params_excitation, range(qubits), s_wires, d_wires, parameter_map, hf_state))
        gradz = get_gradient(circuit_dipole_z)(params_excitation, range(qubits), s_wires, d_wires, parameter_map, hf_state)[0]
        gradz = gradz[len(theta_opt):]

    return gradx, grady, gradz

def hess_diag_approximate(symbols, geometry, charge, basis=None):
    if basis is None:
        basis = 'STO-3G'
    electrons = sum([periodictable.elements.__dict__[symbol].number for symbol in symbols]) - charge
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge, method='pyscf', basis=basis)
    singles, doubles, parameter_map = excitations.spin_adapted_excitations(electrons, qubits)
    mol = qml.qchem.Molecule(symbols, geometry, basis_name=basis, charge=charge)
    scf = qml.qchem.hartree_fock.scf(mol)
    orbital_energies = scf()[0]
    e = np.repeat(orbital_energies, 2) # alpha,beta,alpha,beta
    num_params = max(parameter_map[len(parameter_map)-1][0]) + 1
    hdiag = np.zeros(num_params)
    for idx, (i,a) in enumerate(singles):
        for k, factor in zip(*parameter_map[idx]):
            hdiag[k] += abs(factor*(e[a] - e[i]))
    for idx, (i,j,a,b) in enumerate(doubles):
        for k, factor in zip(*parameter_map[len(singles) + idx]):
            hdiag[k] += abs(factor*(e[a] + e[b] - e[i] - e[j]))
    print(hdiag)
    return hdiag
