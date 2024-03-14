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
import h5py
import copy
from functools import partial

import polarizationsolver
from molecular_hamiltonian import get_molecular_hamiltonian 

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
        H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge, method='pyscf', basis=basis, active_electrons=active_electrons, active_orbitals=active_orbitals)
        hf_filename = f'molecule_pyscf_{basis.strip()}.hdf5'
        electrons = sum([periodictable.elements.__dict__[symbol].number for symbol in symbols]) - charge
        self.active_electrons = active_electrons
        self.active_orbitals = active_orbitals
        self.inactive_electrons = 0
        if active_electrons is not None:
            self.inactive_electrons = electrons - active_electrons
            electrons = active_electrons
        hf_state = qml.qchem.hf_state(electrons, qubits)
        excitations_singlet = excitations.spin_adapted_excitations(electrons, qubits)
        excitations_triplet = excitations.spin_adapted_excitations(electrons, qubits, triplet=True)
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
        def circuit_operator(self, params_ground_state, operator):
            UCCSD(params_ground_state, range(self.qubits), self.excitations_ground_state, self.hf_state)
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
            return qml.expval(operator)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit_exc_operators(self, params_ground_state, params_excitation, operators, triplet=False):
            if triplet:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state, excitations_triplet=self.excitations_triplet)
            else:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state, excitations_singlet=self.excitations_singlet)
            return [qml.expval(operator) for operator in operators]

        @qml.qnode(dev, diff_method="best")
        def circuit_state(self, params_ground_state, params_excitation, triplet=False):
            if triplet:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_singlet, self.hf_state, excitations_triplet=self.excitations_triplet)
            else:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state, excitations_singlet=self.excitations_singlet)
            return qml.state()

        self.H = H
        self.qubits = qubits
        self.electrons = electrons
        self.hf_state = hf_state
        self.excitations_singlet = excitations_singlet
        self.excitations_ground_state = copy.deepcopy(self.excitations_singlet)
        self.excitations_triplet = excitations_triplet
        self.theta = qml.numpy.zeros(len(self.excitations_ground_state))
        self.device = dev
        self.circuit = circuit
        self.circuit_operator = circuit_operator
        self.circuit_operators = circuit_operators
        self.circuit_exc = circuit_exc
        self.circuit_exc_operator = circuit_exc_operator
        self.circuit_exc_operators = circuit_exc_operators
        self.circuit_state = circuit_state
        self.PE = None
        if PE is not None:
            system = polarizationsolver.readers.parser(polarizationsolver.readers.potreader(PE))
            self.PE = system

        atom_str = ''
        for symbol, coord in zip(symbols, geometry):
            atom_str += f'{symbol} {coord[0]} {coord[1]} {coord[2]}; '
            print(atom_str)
        m = pyscf.M(atom=atom_str, basis=basis, charge=charge, unit='bohr')
        mf = pyscf.scf.RHF(m)
        with h5py.File(hf_filename, 'r') as f:
            mf.mo_coeff = f['canonical_orbitals'][()]
            mf.mo_energy = f['orbital_energies'][()]
            print(mf.mo_energy)
        self.m = m
        self.mf = mf
        if PE is not None:
            self.mf = pyscf.solvent.PE(mf, PE)
            self.mf.kernel()

    def rdm1_slow(self, params_ground_state, params_excitation=None, triplet=False):
        rdm1_active = np.zeros((self.qubits//2, self.qubits//2))
        k = 0
        for i in range(self.qubits//2):
            for j in range(i, self.qubits//2):
                fermi = qml.FermiC(2*i)*qml.FermiA(2*j)
                fermi += qml.FermiC(2*i + 1)*qml.FermiA(2*j + 1)
                operator = qml.jordan_wigner(fermi)
                if params_excitation is not None:
                    expval = self.circuit_exc_operator(self, params_ground_state, params_excitation, operator, triplet=triplet)
                else:
                    expval = self.circuit_operator(self, params_ground_state, operator)
                rdm1_active[i, j] = expval
                rdm1_active[j, i] = expval
                k = k + 1
        return rdm1_active

    def rdm1(self, params_ground_state, params_excitation=None, triplet=False):
        rdm1_active = np.zeros((self.qubits//2, self.qubits//2))
        k = 0
        operators = []
        for i in range(self.qubits//2):
            for j in range(i, self.qubits//2):
                fermi = qml.FermiC(2*i)*qml.FermiA(2*j)
                fermi += qml.FermiC(2*i + 1)*qml.FermiA(2*j + 1)
                operator = qml.jordan_wigner(fermi)
                coeffs = []
                obs = []
                for term in operator.terms()[1]:
                    if term.arithmetic_depth > 0:
                        coeff, ob = term.terms()
                        if term.arithmetic_depth == 1:
                            ob = qml.operation.Tensor(*ob)
                        else:
                            ob = qml.operation.Tensor(*ob[0])
                    else:
                        coeff = [1]
                        ob = term
                    coeffs.extend(coeff)
                    obs.append(ob)
                operators.append(qml.Hamiltonian(np.array(coeffs).astype(np.float64), obs).simplify())
        if params_excitation is not None:
            expectation_values = self.circuit_exc_operators(self, params_ground_state, params_excitation, operators, triplet=triplet)
        else:
            expectation_values = self.circuit_operators(self, params_ground_state, operators)
        for i in range(self.qubits//2):
            for j in range(i, self.qubits//2):
                expval = expectation_values[k]
                rdm1_active[i, j] = expval
                rdm1_active[j, i] = expval
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

                H, qubits = get_molecular_hamiltonian(self, self.electrons, self.qubits//2, v_PE=v_PE)
                self.H = H
                self.v_PE_gs = v_PE

            
            energy = self.circuit(self, params)
            #print("<v_PE> = ", np.sum(dm_ao * v_PE))
            grad = get_gradient(self.circuit)(self, params)
            if self.PE:
                #print('E_PE = ', energy + energy_pe_en + E_pol 
                #print("<v_PE> (tot) = ", np.sum(dm_ao*(v_PE)))
                energy += energy_pe_en + E_pol - np.dot((v_ind+v_ind.T).ravel(), dm_ao.ravel())
            #    print("<v_PE> (ind) = ", np.sum(dm_ao*(v_ind)))
            #else:
            print('energy = ', energy)#, E_pol_nuc, E_pol, 0.5*np.dot(self.PE.induced_moments[1].ravel(), F_nuc.ravel()))
            return energy, grad

        res = minimize(energy_and_jac, jac=True, x0=self.theta, method=min_method, tol=1e-12)
        print(res)
        self.theta = res.x

    def hvp(self, v, h=1e-6, scheme='central', triplet=False):
        def grad(x):
            if self.PE:
                # get 1RDM and transform to AO
                dm_mo_gs = self.rdm1(self.theta)
                dm_mo_gs = _make_rdm1_on_mo(dm_mo_gs, self.inactive_electrons//2, self.qubits//2, self.m.nao)
                dm_mo = self.rdm1(self.theta, params_excitation=x)
                dm_mo = _make_rdm1_on_mo(dm_mo, self.inactive_electrons//2, self.qubits//2, self.m.nao)
                delta = dm_mo - dm_mo_gs
                dm_ao = self.mf.mo_coeff @ (dm_mo_gs+0.5*delta) @ self.mf.mo_coeff.T
                # get electric fields from QM
                # get induction contribution
                # modify gas-phase Hamiltonian with v_es + v_ind 
                v_PE = np.zeros_like(dm_ao)
                fakemol = pyscf.gto.fakemol_for_charges(self.PE.coordinates)

                # solve for induced dipoles
                if 1 in (self.PE.active_permanent_multipole_ranks) or (1 in self.PE.active_induced_multipole_ranks):
                    field_integrals = pyscf.df.incore.aux_e2(self.m, fakemol, 'int3c2e_ip1').transpose(1,2,3,0) 
                if 0 in self.PE.active_permanent_multipole_ranks:
                    v_PE += -np.sum(pyscf.df.incore.aux_e2(self.m, fakemol, 'int3c2e')*self.PE.permanent_moments[0], axis=2)
                # dipoles
                # field_integrals could maybe be symmetrized in (0,1) dimension
                if 1 in self.PE.active_permanent_multipole_ranks:
                    v_dip = -np.sum(field_integrals*self.PE.permanent_moments[1], axis=(2,3))
                    v_PE += v_dip + v_dip.T
                # quadrupoles
                if 2 in self.PE.active_permanent_multipole_ranks:
                    # remove trace
                    v_quad = -0.5*np.sum((pyscf.df.incore.aux_e2(self.m, fakemol, 'int3c2e_ipip1') +  pyscf.df.incore.aux_e2(self.m, fakemol, 'int3c2e_ipvip1')).transpose(1,2,3,0) * self.PE.permanent_moments[2].reshape(-1,9), axis=(2,3))
                    v_PE += v_quad + v_quad.T

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

                H, qubits = get_molecular_hamiltonian(self, self.electrons, self.qubits//2, v_PE=v_PE)
                self.H = H
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
        for i in range(v.shape[1]):
            hvp[:, i] = fd_scheme[scheme](grad, h, v[:, i])
        if need_reshape:
            hvp = hvp.reshape(-1)
        return hvp

    def hvp_triplet(self, v, h=1e-6, scheme='central'):
        return self.hvp(v, h=h, scheme=scheme, triplet=True)

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
            elif approach == 'statevector':
                operator_matrix = operator.matrix()

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
            else:
                raise ValueError('Invalid property gradient approach')
        return np.array(operator_gradients).reshape(*out_shape, -1)
