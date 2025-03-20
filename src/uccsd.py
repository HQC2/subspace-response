#!/usr/bin/env python

# based on https://docs.pennylane.ai/en/stable/code/api/pennylane.UCCSD.html

import copy
import time
import functools
import pyscf
import scipy
import numpy as np
import pennylane as qml
from scipy.optimize import minimize
from pennylane._grad import grad as get_gradient
import excitations
from molecular_hamiltonian import get_molecular_hamiltonian, get_PE_hamiltonian
from uccsd_circuits import UCCSD, UCCSD_exc, UCCSD_iH_exc, UCCSD_stateprep
from hashlib import sha1

import polarizationsolver

def read_xyz(filename):
    elements = []
    coordinates = []
    with open(filename, 'r') as f:
        N = int(f.readline())
        f.readline()
        for _ in range(N):
            element, x, y, z, *_ = f.readline().split()
            elements.append(element)
            coordinates.append([float(x), float(y), float(z)])
    return elements, np.array(coordinates)

def _make_rdm1_on_mo(casdm1, ncore, ncas, nmo):
    nocc = ncas + ncore
    dm1 = np.zeros((nmo,nmo))
    idx = np.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1
    return dm1

class uccsd(object):
    def __init__(self, symbols, geometry, charge, basis, active_electrons=None, active_orbitals=None, PE=None):
        atom_str = ''
        for symbol, coord in zip(symbols, geometry):
            atom_str += f'{symbol} {coord[0]} {coord[1]} {coord[2]}; '
        m = pyscf.M(atom=atom_str, basis=basis, charge=charge, unit='bohr')
        mf = pyscf.scf.RHF(m).run()
        self.m = m
        self.mf = mf
        self.PE = None
        if PE is not None:
            system = polarizationsolver.readers.parser(polarizationsolver.readers.potreader(PE))
            self.PE = system
            self.mf = pyscf.solvent.PE(mf, PE)
            self.mf.kernel()
        electrons = sum(m.nelec)
        orbitals = m.nao
        active_electrons = active_electrons if active_electrons else electrons
        active_orbitals = active_orbitals if active_orbitals else orbitals
        self.inactive_electrons = electrons - active_electrons
        self.inactive_orbitals = self.inactive_electrons // 2
        self.active_electrons = active_electrons
        self.active_orbitals = active_orbitals
        H, qubits = get_molecular_hamiltonian(self)
        hf_state = qml.qchem.hf_state(active_electrons, qubits)

        excitations_singlet = excitations.spin_adapted_excitations(active_electrons, qubits)
        excitations_triplet = excitations.spin_adapted_excitations(active_electrons, qubits, triplet=True)
        dev = qml.device("lightning.qubit", wires=qubits)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit(self, params_ground_state):
            UCCSD(params_ground_state, range(self.qubits), self.excitations_ground_state, self.hf_state)
            return qml.expval(self.H)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit_exc(self, params_ground_state, params_excitation, triplet=False):
            if triplet:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state, excitations_triplet=self.excitations_triplet)
            else:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state, excitations_singlet=self.excitations_singlet)
            return qml.expval(self.H)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit_stateprep(self, params_ground_state, statevector):
            UCCSD_stateprep(params_ground_state, statevector, range(self.qubits), self.excitations_ground_state)
            return qml.expval(self.H)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit_operator_stateprep(self, params_ground_state, statevector, operator, triplet=False):
            UCCSD_stateprep(params_ground_state, statevector, range(self.qubits), self.excitations_ground_state)
            return qml.expval(operator)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit_operator(self, params_ground_state, operator):
            UCCSD(params_ground_state, range(self.qubits), self.excitations_ground_state, self.hf_state)
            if isinstance(operator, list):
                return [qml.expval(op) for op in operator]
            else:
                return qml.expval(operator)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit_operators(self, params_ground_state, operators):
            UCCSD(params_ground_state, range(self.qubits), self.excitations_ground_state, self.hf_state)
            return [qml.expval(operator) for operator in operators]

        @qml.qnode(dev, diff_method="adjoint")
        def circuit_exc_operator(self, params_ground_state, params_excitation, operator, triplet=False):
            if triplet:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state, excitations_triplet=self.excitations_triplet)
            else:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state, excitations_singlet=self.excitations_singlet)
            if isinstance(operator, list):
                return [qml.expval(op) for op in operator]
            else:
                return qml.expval(operator)

        @qml.qnode(dev, diff_method="best")
        def circuit_state(self, params_ground_state, params_excitation, triplet=False):
            if triplet:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_singlet, self.hf_state, excitations_triplet=self.excitations_triplet)
            else:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state, excitations_singlet=self.excitations_singlet)
            return qml.state()

        @qml.qnode(dev, diff_method="adjoint")
        def circuit_iH_exc(self, params_ground_state, params_excitation, triplet=False):
            if triplet:
                UCCSD_iH_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state, excitations_triplet=self.excitations_triplet)
            else:
                UCCSD_iH_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state, excitations_singlet=self.excitations_singlet)
            return qml.expval(self.H)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit_iH_exc_operator(self, params_ground_state, params_excitation, operator, triplet=False):
            if triplet:
                UCCSD_iH_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state, excitations_triplet=self.excitations_triplet)
            else:
                UCCSD_iH_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state, excitations_singlet=self.excitations_singlet)
            return qml.expval(operator)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit_operator_stateprep(self, params_ground_state, statevector, operator, triplet=False):
            UCCSD_stateprep(params_ground_state, statevector, range(self.qubits), self.excitations_ground_state)
            if isinstance(operator, list):
                return [qml.expval(op) for op in operator]
            else:
                return qml.expval(operator)

        self.H = H
        self.H_gas = H
        self.qubits = qubits
        self.electrons = electrons
        self.hf_state = hf_state
        self.excitations_singlet = excitations_singlet
        self.excitations_ground_state = copy.deepcopy(self.excitations_singlet)
        self.excitations_triplet = excitations_triplet
        self.excitation_operators_singlet = excitations.excitations_to_operators(excitations_singlet)
        self.theta = qml.numpy.zeros(len(self.excitations_ground_state))
        self.device = dev
        self.circuit = circuit
        self.circuit_operator = circuit_operator
        self.circuit_exc = circuit_exc
        self.circuit_iH_exc = circuit_iH_exc
        self.circuit_iH_exc_operator = circuit_iH_exc_operator
        self.circuit_stateprep = circuit_stateprep
        self.circuit_operator_stateprep = circuit_operator_stateprep
        self.circuit_exc_operator = circuit_exc_operator
        self.circuit_iH_exc_operator = circuit_iH_exc_operator
        self.circuit_state = circuit_state
        self.circuit_operator_stateprep = circuit_operator_stateprep

    def rdm1(self, params_ground_state, params_excitation=None, triplet=False):
        rdm1_active = np.zeros((self.qubits//2, self.qubits//2))
        operators = []
        for i in range(self.qubits//2):
            for j in range(i, self.qubits//2):
                fermi = qml.FermiC(2*i) * qml.FermiA(2*j) 
                fermi += qml.FermiC(2*i+1) * qml.FermiA(2*j+1)
                operator = qml.jordan_wigner(fermi)
                operators.append(operator)
        if params_excitation is not None:
            expvals = self.circuit_exc_operator(self, params_ground_state, params_excitation, operators, triplet=triplet)
        else:
            expvals = self.circuit_operator(self, params_ground_state, operators)
        k = 0
        for i in range(self.qubits//2):
            for j in range(i, self.qubits//2):
                rdm1_active[i,j] = expvals[k].real
                rdm1_active[j,i] = expvals[k].real
                k = k + 1
        return rdm1_active

    def ground_state(self, min_method='slsqp'):
        def energy_and_jac(params):
            params = qml.numpy.array(params)
            energy_pe_en = 0.
            E_pol_nuc = 0.
            E_pol = 0.
            if self.PE:
                # get 1RDM and transform to AO
                dm_mo = self.rdm1(params)
                dm_mo = _make_rdm1_on_mo(dm_mo, self.inactive_electrons//2, self.qubits//2, self.m.nao)
                dm_ao = self.mf.mo_coeff @ dm_mo @ self.mf.mo_coeff.T
                # get electric fields from QM
                # get induction contribution
                # modify gas-phase Hamiltonian with v_es + v_ind 
                v_PE = np.zeros_like(dm_ao)
                fakemol = pyscf.gto.fakemol_for_charges(self.PE.coordinates)
                # charges
                if 1 in (self.PE.active_permanent_multipole_ranks) or (1 in self.PE.active_induced_multipole_ranks):
                    field_integrals = pyscf.df.incore.aux_e2(self.m, fakemol, 'int3c2e_ip1').transpose(1,2,3,0) 
                if 0 in self.PE.active_permanent_multipole_ranks:
                    v_PE += -np.sum(pyscf.df.incore.aux_e2(self.m, fakemol, 'int3c2e')*self.PE.permanent_moments[0], axis=2)
                    energy_pe_en = -(polarizationsolver.fields.field(self.m.atom_coords() - self.PE.coordinates[:,None,:], 0, self.PE.permanent_moments[0], 0) * self.m.atom_charges()).sum()
                # dipoles
                # field_integrals could maybe be symmetrized in (0,1) dimension
                if 1 in self.PE.active_permanent_multipole_ranks:
                    v_dip = -np.sum(field_integrals*self.PE.permanent_moments[1], axis=(2,3))
                    v_PE += v_dip + v_dip.T
                    energy_pe_en += -(polarizationsolver.fields.field(self.m.atom_coords() - self.PE.coordinates[:,None,:], 1, self.PE.permanent_moments[1], 0) * self.m.atom_charges()).sum()
                # quadrupoles
                if 2 in self.PE.active_permanent_multipole_ranks:
                    # remove trace
                    self.PE.permanent_moments[2] -= np.eye(3)[None, :, :] * np.einsum('qii->q', self.PE.permanent_moments[2])[:,None, None]/3
                    v_quad = -0.5*np.sum((pyscf.df.incore.aux_e2(self.m, fakemol, 'int3c2e_ipip1') +  pyscf.df.incore.aux_e2(self.m, fakemol, 'int3c2e_ipvip1')).transpose(1,2,3,0) * self.PE.permanent_moments[2].reshape(-1,9), axis=(2,3))
                    v_PE += v_quad + v_quad.T
                    energy_pe_en += -(polarizationsolver.fields.field(self.m.atom_coords() - self.PE.coordinates[:,None,:], 2, self.PE.permanent_moments[2], 0) * self.m.atom_charges()).sum()

                # solve for induced dipoles
                # rhs field = F_nuc + F_el + F_multipole
                # F_multipole is constructed internally by polarizationsolver
                if 1 in self.PE.active_induced_multipole_ranks:
                    F_nuc = polarizationsolver.fields.field(self.PE.coordinates - self.m.atom_coords()[:,None,:], 0, self.m.atom_charges(), 1)
                    F_rhs = F_nuc + 2*np.einsum('mn,mnpx->px', dm_ao, field_integrals)
                    self.PE.external_field = [[], F_rhs]
                    polarizationsolver.solvers.iterative_solver(self.PE, tol=1e-12)
                    v_ind = -np.sum(field_integrals*self.PE.induced_moments[1], axis=(2,3))
                    v_PE += v_ind + v_ind.T
                    E_pol_nuc =  -0.5*(polarizationsolver.fields.field(self.m.atom_coords() - self.PE.coordinates[:,None,:], 1, self.PE.induced_moments[1], 0) * self.m.atom_charges()).sum()
                    E_pol = self.PE.E_pol

                H_PE, qubits = get_PE_hamiltonian(self, v_PE=v_PE)
                self.H = self.H_gas + H_PE
                self.H_PE_gs = H_PE
    
            energy = self.circuit(self, params)
            grad = get_gradient(self.circuit)(self, params)
            if self.PE:
                energy += energy_pe_en 
                if 1 in self.PE.active_induced_multipole_ranks:
                    energy +=E_pol - np.dot((v_ind+v_ind.T).ravel(), dm_ao.ravel())
            print('energy = ', energy)
            return energy, grad

        res = minimize(energy_and_jac, jac=True, x0=self.theta, method=min_method, tol=1e-12)
        print(res)
        self.theta = res.x


    def hvp(self, v, h=1e-5, scheme='central', triplet=False):
        if triplet:
            excita = self.excitations_triplet
        else:
            excita = self.excitations_singlet
        if self.PE:
            if 1 in self.PE.active_induced_multipole_ranks and not self.PE.GSPOL:
                # get transition density (MO)
                transition_densities = self.transition_density(v, triplet=triplet) # todo skip triplet?
                if self.qubits //2 < self.m.nao:
                    full_transition_densities = np.zeros((v.shape[1], self.m.nao, self.m.nao))
                    full_transition_densities[:, self.inactive_orbitals:self.inactive_orbitals+self.active_orbitals, self.inactive_orbitals:+self.inactive_orbitals+self.active_orbitals] = transition_densities
                    transition_densities = full_transition_densities
                # transform to AO
                transition_densities = self.mf.mo_coeff @ transition_densities @ self.mf.mo_coeff.T
                fakemol = pyscf.gto.fakemol_for_charges(self.PE.coordinates)
                field_integrals = pyscf.df.incore.aux_e2(self.m, fakemol, 'int3c2e_ip1').transpose(1,2,3,0) 
                F_rhs = 2*np.einsum('kmn,mnpx->kpx', transition_densities, field_integrals)
                # get induced dipoles and potential for k'th transition density
                induction_potentials = []
                for k in range(v.shape[1]):
                    self.PE.external_field = [[], F_rhs[k]]
                    self.PE.induced_moments[1] *= 0.
                    polarizationsolver.solvers.iterative_solver(self.PE, tol=1e-12, skip_permanent=True, scheme='GS')
                    v_ind = -np.sum(field_integrals*self.PE.induced_moments[1], axis=(2,3))
                    induction_potentials.append(v_ind+v_ind.T)

        def grad(x):
            return get_gradient(self.circuit_exc, argnum=2)(self, self.theta, x, triplet=triplet)
        fd_scheme = {
            'forward': lambda g, h, v: g(h*v)/h,
            'central': lambda g, h, v: (g(h*v) - g(-h*v))/(2*h),
            '5-point': lambda g, h, v: (g(-2*h*v) - 8*g(-1*h*v) + 8*g(h*v) - g(2*h*v))/(12*h),
            '7-point': lambda g, h, v: (-g(-3*h*v) + 9*g(-2*h*v) - 45*g(-1*h*v) + 45*g(h*v) - 9*g(2*h*v) + g(3*h*v))/(60*h),
            '9-point': lambda g, h, v: (3*g(-4*h*v) - 32*g(-3*h*v) + 168*g(-2*h*v) - 672*g(-1*h*v) + 672*g(h*v) - 168*g(2*h*v) + 32*g(3*h*v) - 3*g(4*h*v))/(840*h),
            '11-point': lambda g, h, v: (-2*g(-5*h*v) + 25*g(-4*h*v) - 150*g(-3*h*v) + 600*g(-2*h*v) - 2100*g(-1*h*v) + 2100*g(h*v) - 600*g(2*h*v) + 150*g(3*h*v) - 25*g(4*h*v) + 2*g(5*h*v))/(2520*h),
        }
        need_reshape = False
        if len(v.shape) == 1:
            v = v.reshape(-1, 1)
            need_reshape = True
        hvp = np.zeros_like(v)
        for k in range(v.shape[1]):
            hvp[:, k] = 2.0*fd_scheme[scheme](grad, h, v[:, k])
            if self.PE:
                if 1 in self.PE.active_induced_multipole_ranks and not self.PE.GSPOL:
                    dynpol_contribution = -self.property_gradient(induction_potentials[k], approach='super')
                    hvp[:, k] += dynpol_contribution
        if need_reshape:
            hvp = hvp.reshape(-1)
        return hvp

    def hvp_new(self, v, triplet=False):
        need_reshape = False
        if len(v.shape) == 1:
            v = v.reshape(-1, 1)
            need_reshape = True
        hvp = np.zeros_like(v)
        e_gr = self.circuit(self, self.theta)
        if triplet:
            excita = self.excitations_triplet
        else:
            excita = self.excitations_singlet
        for k in range(v.shape[1]):
            v_statevector = scipy.sparse.lil_matrix((2**self.qubits, 1))
            for i in range(v.shape[0]):
                v_statevector += v[i,k] * excitations.excitation_to_statevector(self.hf_state, *excita[i])
            # todo: support sparse vector in stateprep
            mvv = self.circuit_exc_stateprep(self, self.theta, v_statevector.toarray().ravel())
            data = []
            for i in range(v.shape[0]):
                i_statevector = excitations.excitation_to_statevector(self.hf_state, *excita[i])
                v_plus_i_statevector = (v_statevector + i_statevector) / np.sqrt(2)
                norm = np.linalg.norm(v_plus_i_statevector.toarray().ravel())
                v_plus_i_statevector /= norm
                mii = self.circuit_exc_stateprep(self, self.theta, i_statevector.toarray().ravel())
                miv = self.circuit_exc_stateprep(self, self.theta, v_plus_i_statevector.toarray().ravel()) * norm**2
                hvp[i, k] = miv - 0.5*mii - 0.5*mvv - v[i,k]*e_gr
        if need_reshape:
            hvp = hvp.reshape(-1)
        return hvp

    def hvp_triplet(self, v, h=1e-6, scheme='central'):
        return self.hvp(v, h=h, scheme=scheme, triplet=True)

    def hvp_new(self, v, triplet=False):
        # hvp via superposition states
        need_reshape = False
        if len(v.shape) == 1:
            v = v.reshape(-1, 1)
            need_reshape = True
        hvp = np.zeros_like(v)
        e_gr = self.circuit(self, self.theta)
        if triplet:
            excita = self.excitations_triplet
        else:
            excita = self.excitations_singlet

        if self.PE:
            if 1 in self.PE.active_induced_multipole_ranks and not self.PE.GSPOL:
                # get transition density (MO)
                transition_densities = self.transition_density(v, triplet=triplet) # todo skip triplet?
                if self.qubits //2 < self.m.nao:
                    full_transition_densities = np.zeros((v.shape[1], self.m.nao, self.m.nao))
                    full_transition_densities[:, self.inactive_orbitals:self.inactive_orbitals+self.active_orbitals, self.inactive_orbitals:+self.inactive_orbitals+self.active_orbitals] = transition_densities
                    transition_densities = full_transition_densities
                # transform to AO
                transition_densities = self.mf.mo_coeff @ transition_densities @ self.mf.mo_coeff.T
                fakemol = pyscf.gto.fakemol_for_charges(self.PE.coordinates)
                field_integrals = pyscf.df.incore.aux_e2(self.m, fakemol, 'int3c2e_ip1').transpose(1,2,3,0) 
                F_rhs = 2*np.einsum('kmn,mnpx->kpx', transition_densities, field_integrals)
                # get induced dipoles and potential for k'th transition density
                induction_potentials = []
                for k in range(v.shape[1]):
                    self.PE.external_field = [[], F_rhs[k]]
                    self.PE.induced_moments[1] *= 0.
                    polarizationsolver.solvers.iterative_solver(self.PE, tol=1e-12, skip_permanent=True, scheme='GS')
                    v_ind = -np.sum(field_integrals*self.PE.induced_moments[1], axis=(2,3))
                    induction_potentials.append(v_ind+v_ind.T)

        for k in range(v.shape[1]):
            v_statevector = scipy.sparse.lil_matrix((2**self.qubits, 1))
            for i in range(v.shape[0]):
                v_statevector += v[i,k] * excitations.excitation_to_statevector(self.hf_state, *excita[i])
            # todo: support sparse vector in stateprep
            mvv = self.circuit_operator_stateprep(self, self.theta, v_statevector.toarray().ravel(), operator=self.H)
            for i in range(v.shape[0]):
                i_statevector = excitations.excitation_to_statevector(self.hf_state, *excita[i])
                v_plus_i_statevector = (v_statevector + i_statevector) / np.sqrt(2)
                norm = np.linalg.norm(v_plus_i_statevector.toarray().ravel())
                v_plus_i_statevector /= norm
                mii = self.circuit_operator_stateprep(self, self.theta, i_statevector.toarray().ravel(), operator=self.H)
                miv = self.circuit_operator_stateprep(self, self.theta, v_plus_i_statevector.toarray().ravel(), operator=self.H) * norm**2
                hvp[i, k] = miv - 0.5*mii - 0.5*mvv - v[i,k]*e_gr
            # pe dynpol contribution
            if self.PE:
                if 1 in self.PE.active_induced_multipole_ranks and not self.PE.GSPOL:
                    dynpol_contribution = -self.property_gradient(induction_potentials[k], approach='super')
                    hvp[:, k] += dynpol_contribution
        if need_reshape:
            hvp = hvp.reshape(-1)
        return hvp

    def transition_density(self, v, triplet=False):
        need_reshape = False
        if len(v.shape) == 1:
            v = v.reshape(-1, 1)
            need_reshape = True
        D_tr = np.zeros((v.shape[1], self.qubits//2, self.qubits//2))
        if triplet:
            excita = self.excitations_triplet
        else:
            excita = self.excitations_singlet
        operators = []
        for i in range(self.qubits//2):
            for j in range(self.qubits//2):
                fermi = qml.FermiC(2*i) * qml.FermiA(2*j) 
                fermi += qml.FermiC(2*i+1) * qml.FermiA(2*j+1)
                operator = qml.jordan_wigner(fermi)
                operators.append(operator)
        D0_expvals = self.circuit_operator(self, self.theta, operators)
        hf_statevector = excitations.occupation_to_statevector(self.hf_state)
        for k in range(v.shape[1]):
            v_statevector = scipy.sparse.lil_matrix((2**self.qubits, 1))
            for i in range(v.shape[0]):
                v_statevector += v[i,k] * excitations.excitation_to_statevector(self.hf_state, *excita[i])
            plus_statevector = (hf_statevector + v_statevector)/np.sqrt(2)
            Dvv_expvals = self.circuit_operator_stateprep(self, self.theta, v_statevector.toarray().ravel(), operator=operators)
            D0v_expvals = self.circuit_operator_stateprep(self, self.theta, plus_statevector.toarray().ravel(), operator=operators)
            unpack_idx = 0
            for i in range(self.qubits//2):
                for j in range(self.qubits//2):
                    D_tr[k,i,j] = D0v_expvals[unpack_idx].real - 0.5*(D0_expvals[unpack_idx].real + Dvv_expvals[unpack_idx].real)
                    unpack_idx += 1 
        if need_reshape:
            # only a single trial-vector
            D_tr = D_tr[0]
        # todo transform active->full space
        return D_tr

    def hess_diag_approximate(self, triplet=False):
        orbital_energies = self.mf.mo_energy
        e = np.repeat(orbital_energies, 2)  # alpha,beta,alpha,beta
        num_params = len(self.excitations_singlet)
        excitations = self.excitations_singlet
        if triplet:
            num_params = len(self.excitations_triplet)
            excitations = self.excitations_triplet
        hdiag = np.zeros(num_params)
        for k, (excitation_group, weights) in enumerate(excitations):
            first = excitation_group[0]
            occ_indices = first[:len(first)//2]
            vir_indices = first[len(first)//2:]
            hdiag[k] = sum(e[vir_indices]) - sum(e[occ_indices])
        return hdiag

    def expectation_value(self, integral):
        if isinstance(integral, str):
            ao_integrals = self.m.intor(integral)
        elif isinstance(integral, np.ndarray):
            ao_integrals = integral
        else:
            raise ValueError('Integral should be provided either as a string to evaluate with pyscf m.intor or as a plain numpy array (AO basis).')
        out_shape = ao_integrals.shape[:-2]
        ao_integrals = ao_integrals.reshape(-1, self.m.nao, self.m.nao)
        mo_integrals = np.einsum('uj,xuv,vi->xij', self.mf.mo_coeff, ao_integrals, self.mf.mo_coeff)
        expectation_values = []
        for component in mo_integrals:
            # skip null operator
            if np.allclose(component, 0.0):
                expectation_values.append(0.0)
                print('Skipping null operator')
                continue

            operator = 0.0
            I = self.inactive_electrons//2
            for p in range(self.qubits//2):
                for q in range(self.qubits//2):
                    operator += component[I + p, I + q]*qml.FermiC(2*p)*qml.FermiA(2*q)
                    operator += component[I + p, I + q]*qml.FermiC(2*p + 1)*qml.FermiA(2*q + 1)
            # with casci-style active space, add contribution due to doubly occupied MOs
            operator = qml.jordan_wigner(operator)
            term = 0.
            if self.active_electrons is not None:
                num_inactive = self.inactive_electrons//2
                for i in range(num_inactive):
                    term += 2*component[i, i]
            expectation_values.append(self.circuit_operator(self, self.theta, operator) + term)
        return np.array(expectation_values).reshape(out_shape)

    def property_gradient(self, integral, approach='derivative', triplet=False):
        if isinstance(integral, str):
            ao_integrals = self.m.intor(integral)
        elif isinstance(integral, np.ndarray):
            ao_integrals = integral
        else:
            raise ValueError('Integral should be provided either as a string to evaluate with pyscf m.intor or as a plain numpy array (AO basis).')
        out_shape = ao_integrals.shape[:-2]
        ao_integrals = ao_integrals.reshape(-1, self.m.nao, self.m.nao)
        mo_integrals = np.einsum('uj,xuv,vi->xij', self.mf.mo_coeff, ao_integrals, self.mf.mo_coeff)
        operator_gradients = []

        if triplet:
            parameter_excitation = qml.numpy.zeros(len(self.excitations_triplet))
        else:
            parameter_excitation = qml.numpy.zeros(len(self.excitations_singlet))

        for component in mo_integrals:
            # skip null operator
            if np.allclose(component, 0.0):
                operator_gradients.append(np.zeros_like(parameter_excitation))
                print('Null integral skipped in property gradient')
                continue

            operator = 0.0
            sign = -1 if triplet else 1
            I = self.inactive_electrons//2
            for p in range(self.qubits//2):
                for q in range(self.qubits//2):
                    operator += component[I + p, I + q]*qml.FermiC(2*p)*qml.FermiA(2*q)
                    operator += sign*component[I + p, I + q]*qml.FermiC(2*p + 1)*qml.FermiA(2*q + 1)
            operator = qml.jordan_wigner(operator)

            if approach == 'derivative':
                operator_gradient = get_gradient(self.circuit_exc_operator, argnum=2)(self, self.theta, parameter_excitation, operator, triplet=triplet)
                operator_gradients.append(operator_gradient)
            elif approach == 'iH-derivative':
                operator_gradient = get_gradient(self.circuit_iH_exc_operator, argnum=2)(self, self.theta, parameter_excitation, 1j*operator, triplet=triplet)
                operator_gradients.append(operator_gradient)
            elif approach == 'statevector':
                operator_matrix = operator.matrix(wire_order=range(self.qubits))

                operator_gradient = np.zeros_like(parameter_excitation)
                state_0 = self.circuit_state(self, self.theta, parameter_excitation, triplet=triplet)
                h = 1e-3
                for i in range(len(parameter_excitation)):
                    parameter_excitation[i] = h
                    state_plus = self.circuit_state(self, self.theta, parameter_excitation, triplet=triplet)
                    parameter_excitation[i] = -h
                    state_minus = self.circuit_state(self, self.theta, parameter_excitation, triplet=triplet)
                    parameter_excitation[i] = 0.
                    diff_state = (state_plus - state_minus)/(2*h)
                    operator_gradient[i] = 2*diff_state.conj() @ operator_matrix @ state_0
                operator_gradients.append(operator_gradient)
            elif approach == 'super':
                # <0|O|i>
                if triplet:
                    excita = self.excitations_triplet
                else:
                    excita = self.excitations_singlet
                hf_statevector = excitations.occupation_to_statevector(self.hf_state)
                hf_expval = self.circuit_operator(self, self.theta, operator)
                operator_gradient = np.zeros_like(parameter_excitation)
                for i in range(len(parameter_excitation)):
                    i_statevector = excitations.excitation_to_statevector(self.hf_state, *excita[i])
                    plus_statevector = (hf_statevector + i_statevector)/np.sqrt(2)
                    i_expval = self.circuit_operator_stateprep(self, self.theta, i_statevector.toarray().ravel(), operator=operator)
                    plus_expval = self.circuit_operator_stateprep(self, self.theta, plus_statevector.toarray().ravel(), operator=operator)
                    operator_gradient[i] = (plus_expval - 0.5 * (i_expval + hf_expval)).real
                # sign?
                operator_gradients.append(-operator_gradient)
            elif approach == 'super-imag':
                # <0|O|i> as above, but for imaginary operator
                if triplet:
                    excita = self.excitations_triplet
                else:
                    excita = self.excitations_singlet
                hf_statevector = excitations.occupation_to_statevector(self.hf_state)
                hf_expval = self.circuit_operator(self, self.theta, operator)
                operator_gradient = np.zeros_like(parameter_excitation)
                for i in range(len(parameter_excitation)):
                    i_statevector = excitations.excitation_to_statevector(self.hf_state, *excita[i])
                    plus_statevector = (hf_statevector + 1j*i_statevector)/np.sqrt(2)
                    i_expval = self.circuit_operator_stateprep(self, self.theta, i_statevector.toarray().ravel(), operator=operator)
                    plus_expval = self.circuit_operator_stateprep(self, self.theta, plus_statevector.toarray().ravel(), operator=operator)
                    operator_gradient[i] = plus_expval - 0.5 * (i_expval + hf_expval)
                # sign might need extra fixing here
                operator_gradients.append(-operator_gradient)
            else:
                raise ValueError('Invalid property gradient approach')
        return np.array(operator_gradients).reshape(*out_shape, -1)

    def V2_contraction(self, integral, I, I_dag, J, J_dag, triplet=False, termcache={}):
        if self.PE is not None:
            raise NotImplementedError
        if triplet:
            # todo
            raise NotImplementedError
            excitation_operators = self.excitation_operators_triplet
        else:
            excitation_operators = self.excitation_operators_singlet
        if isinstance(integral, str):
            ao_integral = self.m.intor(integral)
        elif isinstance(integral, np.ndarray):
            ao_integral = integral
        else:
            raise ValueError('Integral should be provided either as a string to evaluate with pyscf m.intor or as a plain numpy array (AO basis).')
        if len(ao_integral.shape) != 2:
            raise ValueError('Integral must have just a single component (dimension (nao,nao)).')
        mo_integral = np.einsum('uj,uv,vi->ij', self.mf.mo_coeff, ao_integral, self.mf.mo_coeff)

        # skip null operator
        if np.allclose(mo_integral, 0.0):
            return 0.0

        operator = 0.0
        sign = -1 if triplet else 1
        for p in range(self.qubits//2):
            for q in range(self.qubits//2):
                operator += mo_integral[self.inactive_electrons//2 + p, self.inactive_electrons//2 + q]*qml.FermiC(2*p)*qml.FermiA(2*q)
                operator += sign*mo_integral[self.inactive_electrons//2 + p, self.inactive_electrons//2 + q]*qml.FermiC(2*p + 1)*qml.FermiA(2*q + 1)
        operator = qml.jordan_wigner(operator)

        op_I = qml.matrix(sum([I[i] * excitation_operators[i] for i in range(len(excitation_operators))]).simplify(), wire_order=range(self.qubits))
        op_I_dag = qml.matrix(sum([I_dag[i] * excitation_operators[i] for i in range(len(excitation_operators))]).simplify(), wire_order=range(self.qubits))
        op_J = qml.matrix(sum([J[i] * excitation_operators[i] for i in range(len(excitation_operators))]).simplify(), wire_order=range(self.qubits))
        op_J_dag = qml.matrix(sum([J_dag[i] * excitation_operators[i] for i in range(len(excitation_operators))]).simplify(), wire_order=range(self.qubits))
        integral_hash = sha1(mo_integral.view(np.uint8)).hexdigest()

        def term(left_ops, right_ops, operator, cache=termcache):
            # |R> = right_op |0>
            # |L> = left_op |0>
            # compute <L|U'O U|R>
            def lazycalc(f, *args, cache=termcache):
                # look only at statevec arg
                key = sha1(np.round(args[2], 6).view(np.uint8)).hexdigest() + integral_hash
                if not key in cache:
                    cache[key] = f(*args)
                return cache[key]
            hf_statevector = np.zeros(2**len(self.hf_state), dtype=np.complex128)
            index = np.sum((self.hf_state)*2**(np.arange(self.qubits)[::-1]))
            hf_statevector[index] = 1

            L_statevec = functools.reduce(np.dot, left_ops + [hf_statevector])
            R_statevec = functools.reduce(np.dot, right_ops + [hf_statevector])
            plus_statevec = L_statevec + R_statevec
            L_norm = np.linalg.norm(L_statevec)
            R_norm = np.linalg.norm(R_statevec)
            plus_norm = np.linalg.norm(plus_statevec)
            L_expval = L_norm**2*lazycalc(self.circuit_operator_stateprep, self, self.theta, L_statevec/L_norm, operator) if L_norm > 1e-9 else 0.
            R_expval = R_norm**2*lazycalc(self.circuit_operator_stateprep, self, self.theta, R_statevec/R_norm, operator) if R_norm > 1e-9 else 0.
            plus_expval = plus_norm**2*lazycalc(self.circuit_operator_stateprep, self, self.theta, plus_statevec/plus_norm, operator) if plus_norm > 1e-9 else 0.
            return 0.5*(plus_expval - L_expval - R_expval)

        # (dagger,dagger) term
        # -<Psi|I'J'O|Psi>
        total = 0.0
        total -= term([op_J_dag, op_I_dag], [], operator)
        
        # (dagger,.) term
        # <Psi|I'OJ - I'JO|Psi>
        total += term([op_I_dag], [op_J], operator) - term([op_J.T.conj(), op_I_dag], [], operator)

        # (.,dagger) term
        # <Psi|J'OI - OJ'I|Psi>
        total += term([op_J_dag], [op_I], operator) - term([], [op_J_dag.T.conj(), op_I], operator)

        # (.,.) term
        # -<Psi|OJI|Psi>
        total -= term([], [op_J,op_I], operator)
        return total

    def E3_contraction(self, I, I_dag, J, J_dag, K, K_dag, triplet=False, termcache={}):
        if self.PE is not None:
            raise NotImplementedError
        if triplet:
            # todo
            raise NotImplementedError
            excitation_operators = self.excitation_operators_triplet
        else:
            excitation_operators = self.excitation_operators_singlet
        op_I = qml.matrix(sum([I[i] * excitation_operators[i] for i in range(len(excitation_operators))]), wire_order=range(self.qubits))
        op_I_dag = qml.matrix(sum([I_dag[i] * excitation_operators[i] for i in range(len(excitation_operators))]), wire_order=range(self.qubits))
        op_J = qml.matrix(sum([J[i] * excitation_operators[i] for i in range(len(excitation_operators))]), wire_order=range(self.qubits))
        op_J_dag = qml.matrix(sum([J_dag[i] * excitation_operators[i] for i in range(len(excitation_operators))]), wire_order=range(self.qubits))
        op_K = qml.matrix(sum([K[i] * excitation_operators[i] for i in range(len(excitation_operators))]), wire_order=range(self.qubits))
        op_K_dag = qml.matrix(sum([K_dag[i] * excitation_operators[i] for i in range(len(excitation_operators))]), wire_order=range(self.qubits))

        def term(left_ops, right_ops, operator):
            # |R> = right_op |0>
            # |L> = left_op |0>
            # compute <L|U'O U|R>
            def lazycalc(f, *args, cache=termcache):
                # look only at statevec arg ([2])
                key = sha1(np.round(args[2], 6).view(np.uint8)).hexdigest()
                if not key in cache:
                    cache[key] = f(*args)
                return cache[key]
            hf_statevector = np.zeros(2**len(self.hf_state), dtype=np.complex128)
            index = np.sum((self.hf_state)*2**(np.arange(self.qubits)[::-1]))
            hf_statevector[index] = 1

            L_statevec = functools.reduce(np.dot, left_ops + [hf_statevector])
            R_statevec = functools.reduce(np.dot, right_ops + [hf_statevector])
            plus_statevec = L_statevec + R_statevec
            L_norm = np.linalg.norm(L_statevec)
            R_norm = np.linalg.norm(R_statevec)
            plus_norm = np.linalg.norm(plus_statevec)
            L_expval = L_norm**2*lazycalc(self.circuit_operator_stateprep, self, self.theta, L_statevec/L_norm, operator) if L_norm > 1e-9 else 0.
            R_expval = R_norm**2*lazycalc(self.circuit_operator_stateprep, self, self.theta, R_statevec/R_norm, operator) if R_norm > 1e-9 else 0.
            plus_expval = plus_norm**2*lazycalc(self.circuit_operator_stateprep, self, self.theta, plus_statevec/plus_norm, operator) if plus_norm > 1e-9 else 0.
            return 0.5*(plus_expval - L_expval - R_expval)
        
        total = 0.0
        # (dagger,dagger,dagger) <Psi|I'J'K'H|Psi>
        total += term([op_K_dag,op_J_dag,op_I_dag], [], self.H)
        # (.,dagger,dagger) <Psi|-J'K'HI + J'HK'I + K'HJ'I - HK'J'I |Psi>
        total -= term([op_K_dag,op_J_dag], [op_I], self.H)
        total += term([op_J_dag], [op_K_dag.T.conj(),op_I], self.H)
        total += term([op_K_dag], [op_J_dag.T.conj(),op_I], self.H)
        total -= term([],[op_K_dag.T.conj(),op_J_dag.T.conj(),op_I], self.H)
        # (dagger,.,dagger) <Psi|I'JK'H - I'K'HJ + I'HK'J|Psi>
        total += term([op_K_dag,op_J.T.conj(),op_I_dag], [], self.H)
        total -= term([op_K_dag,op_I_dag], [op_J], self.H)
        total += term([op_I_dag], [op_K_dag.T.conj(),op_J], self.H)
        # (dagger,dagger,.) <Psi|I'J'KH - I'J'HK|Psi>
        total += term([op_K.T.conj(),op_J_dag,op_I_dag], [], self.H)
        total -= term([op_J_dag,op_I_dag], [op_K], self.H)
        # (.,.,dagger) <Psi|K'HJI - HK'JI|Psi>
        total += term([op_K_dag], [op_J,op_I], self.H)
        total -= term([], [op_K_dag.T.conj(),op_J,op_I], self.H)
        # (.,dagger,.) <Psi|-J'KHI + J'HKI - HKJ'I|Psi>
        total -= term([op_K.T.conj(),op_J_dag], [op_I], self.H)
        total += term([op_J_dag], [op_K,op_I], self.H)
        total -= term([], [op_K,op_J_dag.T.conj(),op_I], self.H)
        # (dagger, ., .) <Psi|I'JKH - I'JHK - I'KHJ + I'HKJ|Psi>
        total += term([op_K.T.conj(),op_J.T.conj(),op_I_dag], [], self.H)
        total -= term([op_J.T.conj(),op_I_dag], [op_K], self.H)
        total -= term([op_K.T.conj(),op_I_dag], [op_J], self.H)
        total += term([op_I_dag], [op_K,op_J], self.H)
        # (.,.,.) <Psi|-HKJI|Psi>
        total -= term([], [op_K,op_J,op_I], self.H)
        return 0.5*total

    def S3_contraction(self, I, I_dag, J, J_dag, K, K_dag, triplet=False):
        if triplet:
            excitations = self.excitations_triplet
        else:
            excitations = self.excitations_singlet

        excitation_rank = np.array([len(excitation[0][0])//2 for excitation in excitations]) 
        singles = np.where(excitation_rank == 1)[0]
        doubles = np.where(excitation_rank == 2)[0]

        result = 0.0
        for D_idx in doubles:
            for (D_exc, wD) in zip(*excitations[D_idx]):
                i,j,a,b = D_exc
                for S1_idx in singles:
                    for (S1_exc, wS1) in zip(*excitations[S1_idx]):
                        k1, c1 = S1_exc
                        # (ia,jb), (ib,ja), (ja,ib), (jb,ia)
                        # first index is ia,ib,ja, or jb
                        if (k1==i or k1==j) and (c1==a or c1==b):
                            k, c = set([i,j,a,b]).difference([k1,c1])
                            for S2_idx in singles:
                                for (S2_exc, wS2) in zip(*excitations[S2_idx]):
                                    k2, c2 = S2_exc
                                    if (k2==k) and (c2==c):
                                        #  I'J'K
                                        # -I'K'J
                                        # -J'KI
                                        #  K'JI
                                        phase = 1 if k1>k2 else -1
                                        phase *= 1 if c1>c2 else -1
                                        #print(phase, k1>k2, c1>c2)
                                        w = phase*wS1*wS2*wD
                                        result += w*I_dag[S1_idx]*J_dag[S2_idx]*K[D_idx]
                                        result -= w*I_dag[S1_idx]*K_dag[S2_idx]*J[D_idx]
                                        result -= w*J_dag[D_idx]*K[S1_idx]*I[S2_idx]
                                        result += w*K_dag[D_idx]*I[S1_idx]*J[S2_idx]
        return -0.5*result
