#!/usr/bin/env python

import pennylane as qml
import numpy as np
import uccsd
from molecular_hamiltonian import get_molecular_hamiltonian 
from scipy.optimize import minimize
import scipy
from pennylane._grad import grad as get_gradient
from functools import partial
import pyscf
from pyscf.mcscf.addons import _make_rdm12_on_mo

def Ukappa(kappa_packed, occupied, active, virtual):
    norb = len(occupied) + len(active) + len(virtual)
    kappa = np.zeros((norb, norb))

    k = 0
    # occupied-active
    for i in occupied:
        for j in active:
            kappa[i,j] = kappa_packed[k]
            kappa[j,i] = -kappa_packed[k]
            k = k + 1
    # occupied-virtual
    for i in occupied:
        for j in virtual:
            kappa[i,j] = kappa_packed[k]
            kappa[j,i] = -kappa_packed[k]
            k = k + 1
    # active-virtual
    for i in active:
        for j in virtual:
            kappa[i,j] = kappa_packed[k]
            kappa[j,i] = -kappa_packed[k]
            k = k + 1
    return scipy.linalg.expm(-kappa)

class oo_uccsd(uccsd.uccsd):
    def __init__(self, symbols, geometry, charge, basis, active_electrons=None, active_orbitals=None):
        super().__init__(symbols, geometry, charge, basis, active_electrons=active_electrons, active_orbitals=active_orbitals)
        
        mult = 1
        self.active_electrons = active_electrons
        self.active_orbitals = active_orbitals
        orbital_indices = range(len(self.mf.mo_coeff))
        occupied_indices, active_indices = qml.qchem.active_space(sum(self.mf.mol.nelec), self.mf.mol.nao, mult, active_electrons, active_orbitals)
        virtual_indices = sorted(set(orbital_indices) - set(occupied_indices) - set(active_indices))
        n_IA = len(occupied_indices) * len(active_indices)
        n_IV = len(occupied_indices) * len(virtual_indices)
        n_AV = len(active_indices) * len(virtual_indices)
        n_kappa = n_IA + n_IV + n_AV
        self.h1e_mo = None
        self.h2e_mo = None

        H, qubits = get_molecular_hamiltonian(self, active_electrons, active_orbitals)
        self.H = H
        self.n_kappa = n_kappa
        self.orbital_indices = orbital_indices
        self.occupied_indices = occupied_indices
        self.active_indices = active_indices
        self.virtual_indices = virtual_indices
        self.n_active = len(active_indices)
        self.C_HF = np.copy(self.mf.mo_coeff)

    def rdm1(self):
        rdm1_active = np.zeros((self.n_active, self.n_active))
        k = 0
        for i in range(self.n_active):
            for j in range(i, self.n_active):
                fermi = qml.FermiC(2*i) * qml.FermiA(2*j) 
                fermi += qml.FermiC(2*i+1) * qml.FermiA(2*j+1)
                operator = qml.jordan_wigner(fermi)
                expval = self.circuit_operator(self, self.theta, operator)
                rdm1_active[i,j] = expval
                rdm1_active[j,i] = expval
                k = k + 1
        return rdm1_active


    def rdm2(self):
        rdm2_active = np.zeros((self.n_active, self.n_active, self.n_active, self.n_active))
        for p in range(self.n_active):
            for q in range(self.n_active):
                for r in range(self.n_active):
                    for s in range(self.n_active):
                        fermi = qml.FermiC(2*p) * qml.FermiC(2*r) * qml.FermiA(2*s) * qml.FermiA(2*q)
                        fermi += qml.FermiC(2*p) * qml.FermiC(2*r+1) * qml.FermiA(2*s+1)  * qml.FermiA(2*q)
                        fermi += qml.FermiC(2*p+1) * qml.FermiC(2*r) * qml.FermiA(2*s) * qml.FermiA(2*q+1)
                        fermi += qml.FermiC(2*p+1) * qml.FermiC(2*r+1) * qml.FermiA(2*s+1) * qml.FermiA(2*q+1)
                        operator = qml.jordan_wigner(fermi)
                        rdm2_active[p,q,r,s] = self.circuit_operator(self, self.theta, operator)
        return rdm2_active

    def orbital_gradient(self, rdm1_active, rdm2_active):
        rdm1_mo, rdm2_mo = _make_rdm12_on_mo(rdm1_active, rdm2_active, len(self.occupied_indices), len(self.active_indices), len(self.orbital_indices))
        X = np.einsum('pr,qr->pq', rdm1_mo, self.h1e_mo) 
        X += np.einsum('prst,qrst->pq', rdm2_mo, self.h2e_mo)

        gradient = np.zeros(self.n_kappa)
        k = 0
        # occupied-active
        for i in self.occupied_indices:
            for j in self.active_indices:
                gradient[k] = 2*(X[i,j] - X[j,i])
                k = k + 1
        # occupied-virtual
        for i in self.occupied_indices:
            for j in self.virtual_indices:
                gradient[k] = 2*(X[i,j] - X[j,i])
                k = k + 1
        # active-virtual
        for i in self.active_indices:
            for j in self.virtual_indices:
                gradient[k] = 2*(X[i,j] - X[j,i])
                k = k + 1
        return gradient

    def ground_state(self, min_method='slsqp'):
        def energy_and_gradient(parameters):
            parameters = qml.numpy.array(parameters)
            kappa = parameters[:self.n_kappa]
            theta = qml.numpy.array(parameters[self.n_kappa:])
            self.theta = theta
            #print('theta=', theta, 'kappa=', kappa)
            
            # transform MOs + and update molecular hamiltonian
            U = Ukappa(kappa, self.occupied_indices, self.active_indices, self.virtual_indices)
            self.mf.mo_coeff = self.C_HF @ U
            H, qubits = get_molecular_hamiltonian(self, self.active_electrons, self.active_orbitals)
            self.H = H

            energy = self.circuit(self, theta)
            print('energy = ', energy, energy--227.910544578929)

            grad_ci = get_gradient(self.circuit)(self, theta)
            rdm1_active = self.rdm1()
            rdm2_active = self.rdm2()
            grad_orb = self.orbital_gradient(rdm1_active, rdm2_active)
            gradient = np.hstack([grad_orb, grad_ci])
            return energy, gradient
        
        self.mf.run()
        mp = pyscf.mp.MP2(self.mf).run()
        noons, natorbs = pyscf.mcscf.addons.make_natural_orbitals(mp)
        self.C_HF = natorbs
        x0 = np.hstack([np.zeros(self.n_kappa), self.theta])
        res = minimize(energy_and_gradient, jac=True, x0=x0, method=min_method, tol=1e-12, options={'maxiter':1000})
        print(res)
        self.theta = res.x

symbols = list('OHH')
geometry = qml.numpy.array([
[0.0,  0.0         ,  0.1035174918],
[0.0,  0.7955612117, -0.4640237459],
[0.0, -0.7955612117, -0.4640237459],
]
                 , requires_grad=False) * 1.88973
basis = 'cc-pVDZ'
charge = 0

oo = oo_uccsd(symbols, geometry, charge, basis, active_electrons=2, active_orbitals=2)
oo.ground_state()
