#!/usr/bin/env python

# based on https://docs.pennylane.ai/en/stable/code/api/pennylane.UCCSD.html

import pennylane as qml
import numpy as np
from pennylane._grad import grad as get_gradient
from scipy.optimize import minimize
import periodictable
import functools
from uccsd_exc import UCCSD_exc

def uccsd_ground_state(symbols, geometry, charge):
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge, method='pyscf')

    electrons = sum([periodictable.elements.__dict__[symbol].number for symbol in symbols]) - charge

    hf_state = qml.qchem.hf_state(electrons, qubits)
    singles, doubles = qml.qchem.excitations(electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    dev = qml.device("lightning.qubit", wires=qubits)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params, wires, s_wires, d_wires, hf_state):
        qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
        return qml.expval(H)

    def energy(params):
        params = qml.numpy.array(params)
        energy = circuit(params, range(qubits), s_wires, d_wires, hf_state)
        print('energy = ', energy)
        return energy

    def jac(params):
        params = qml.numpy.array(params)
        grad = get_gradient(circuit)(params, range(qubits), s_wires, d_wires, hf_state)[0]
        return grad

    params = qml.numpy.zeros(len(singles) + len(doubles))
    res = minimize(energy, jac=jac, x0=params, method='slsqp', tol=1e-12)
    theta_opt = res.x

    return theta_opt

def uccsd_hvp(symbols, geometry, charge, theta_opt):
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge, method='pyscf')

    electrons = sum([periodictable.elements.__dict__[symbol].number for symbol in symbols]) - charge

    hf_state = qml.qchem.hf_state(electrons, qubits)
    singles, doubles = qml.qchem.excitations(electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    dev = qml.device("lightning.qubit", wires=qubits)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit_sd(params, wires, s_wires, d_wires, singles, doubles, hf_state):
        UCCSD_exc(params, wires, s_wires, d_wires, singles, doubles, hf_state)
        return qml.expval(H)

    def jac_sd(params_excitation):
        params_excitation = qml.numpy.array(params_excitation)
        grad = get_gradient(circuit_sd)(params_excitation, range(qubits), s_wires, d_wires, singles, doubles, hf_state)[0]
        return grad

    def jac_exc(v, theta_opt=theta_opt, full_grad=jac_sd):
        # gradient on SingleExcitation + DoubleExcitation parameter part
        params = qml.numpy.zeros(len(theta_opt) + len(v))
        params[:len(theta_opt)] = theta_opt
        params[len(theta_opt):] = v
        fg = full_grad(params)
        grad = fg[len(theta_opt):]
        return grad

    def hessian_vector_product(v, grad, theta, h=1e-6):
        hvp = np.zeros_like(v)
        if len(v.shape) == 1:
            hvp = (grad(theta+h*v) - grad(theta-h*v)) / (2*h)
        elif len(v.shape) == 2:
            for i in range(v.shape[1]):
                hvp[:, i] =  (grad(theta+h*v[:,i]) - grad(theta-h*v[:,i])) / (2*h)
        else:
            raise ValueError
        return hvp

    dim = 1*(len(singles) + len(doubles))
    hvp = functools.partial(hessian_vector_product, grad=jac_exc, theta=qml.numpy.zeros(dim))
    return hvp

def uccsd_dipole_property_gradient(symbols, geometry, charge, theta_opt):
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge, method='pyscf')

    electrons = sum([periodictable.elements.__dict__[symbol].number for symbol in symbols]) - charge

    hf_state = qml.qchem.hf_state(electrons, qubits)
    singles, doubles = qml.qchem.excitations(electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    dev = qml.device("lightning.qubit", wires=qubits)
    mol = qml.qchem.Molecule(symbols, geometry)
    dipole_x = qml.qchem.qubit_observable(qml.qchem.fermionic_observable(np.array([0.0]), qml.qchem.dipole_integrals(mol)()[1][0]))
    dipole_y = qml.qchem.qubit_observable(qml.qchem.fermionic_observable(np.array([0.0]), qml.qchem.dipole_integrals(mol)()[1][1]))
    dipole_z = qml.qchem.qubit_observable(qml.qchem.fermionic_observable(np.array([0.0]), qml.qchem.dipole_integrals(mol)()[1][2]))

    @qml.qnode(dev, diff_method="adjoint")
    def circuit_dipole_x(params, wires, s_wires, d_wires, hf_state):
        UCCSD_exc(params, wires, s_wires, d_wires, singles, doubles, hf_state)
        return qml.expval(dipole_x)
    @qml.qnode(dev, diff_method="adjoint")
    def circuit_dipole_y(params, wires, s_wires, d_wires, hf_state):
        UCCSD_exc(params, wires, s_wires, d_wires, singles, doubles, hf_state)
        return qml.expval(dipole_y)
    @qml.qnode(dev, diff_method="adjoint")
    def circuit_dipole_z(params, wires, s_wires, d_wires, hf_state):
        UCCSD_exc(params, wires, s_wires, d_wires, singles, doubles, hf_state)
        return qml.expval(dipole_z)

    params_excitation = qml.numpy.zeros(len(theta_opt) + len(singles) + len(doubles))
    params_excitation[:len(theta_opt)] = theta_opt
    gradx = get_gradient(circuit_dipole_x)(params_excitation, range(qubits), s_wires, d_wires, hf_state)[0]
    grady = get_gradient(circuit_dipole_y)(params_excitation, range(qubits), s_wires, d_wires, hf_state)[0]
    gradz = get_gradient(circuit_dipole_z)(params_excitation, range(qubits), s_wires, d_wires, hf_state)[0]
    # we want gradients on the excitation operator part
    gradx = gradx[len(theta_opt):]
    grady = grady[len(theta_opt):]
    gradz = gradz[len(theta_opt):]

    return gradx, grady, gradz

def hess_diag_approximate(symbols, geometry, charge):
    electrons = sum([periodictable.elements.__dict__[symbol].number for symbol in symbols]) - charge
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge, method='pyscf')
    singles, doubles = qml.qchem.excitations(electrons, qubits)
    mol = qml.qchem.Molecule(symbols, geometry)
    scf = qml.qchem.hartree_fock.scf(mol)
    orbital_energies = scf()[0]
    orbital_energies, singles, doubles
    e = np.repeat(orbital_energies, 2) # alpha,beta,alpha,beta
    hdiag = np.zeros(len(singles) + len(doubles))
    k = 0
    for i,a in singles:
        hdiag[k] = (e[a] - e[i])
        k = k + 1
    for i,j,a,b in doubles:
        hdiag[k] = e[a] + e[b] - e[i] - e[j]
        k = k + 1
    return hdiag
