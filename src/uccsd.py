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
import solvers

class uccsd(object):
    def __init__(self, symbols, geometry, charge, basis, active_electrons=None, active_orbitals=None):
        H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=charge, method='pyscf', basis=basis,
                                                    active_electrons=active_electrons, active_orbitals=active_orbitals)
        hf_filename = f'molecule_pyscf_{basis.strip()}.hdf5'
        electrons = sum([periodictable.elements.__dict__[symbol].number for symbol in symbols]) - charge
        if active_electrons is not None:
            electrons = active_electrons
        hf_state = qml.qchem.hf_state(electrons, qubits)
        hf_state.requires_grad = False
        excitations_singlet = excitations.spin_adapted_excitations(electrons, qubits)
        excitations_triplet = excitations.spin_adapted_excitations(electrons, qubits, triplet=True)
        dev = qml.device("lightning.qubit", wires=qubits)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit(self, params_ground_state):
            UCCSD(params_ground_state, range(self.qubits), self.excitations_ground_state, self.hf_state)
            return qml.expval(H)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit_exc(self, params_ground_state, params_excitation, triplet=False):
            if triplet:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state,
                          excitations_triplet=self.excitations_triplet)
            else:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state,
                          excitations_singlet=self.excitations_singlet)
            return qml.expval(H)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit_operator(self, params_ground_state, operator):
            UCCSD(params_ground_state, range(self.qubits), self.excitations_ground_state, self.hf_state)
            return qml.expval(operator)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit_exc_operator(self, params_ground_state, params_excitation, operator, triplet=False):
            if triplet:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state,
                  excitations_triplet=self.excitations_triplet)
            else:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state,
                          excitations_singlet=self.excitations_singlet)
            return qml.expval(operator)

        @qml.qnode(dev, diff_method="best")
        def circuit_state(self, params_ground_state, params_excitation, triplet=False):
            if triplet:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_singlet, self.hf_state,
                  excitations_triplet=self.excitations_triplet)
            else:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state,
                          excitations_singlet=self.excitations_singlet)
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
        self.circuit_exc = circuit_exc
        self.circuit_exc_operator = circuit_exc_operator
        self.circuit_state = circuit_state

        atom_str = ''
        for symbol, coord in zip(symbols, geometry):
            atom_str += f'{symbol} {coord[0]} {coord[1]} {coord[2]}; '
            print(atom_str)
        m = pyscf.M(atom=atom_str, basis=basis, charge=charge, unit='bohr')
        mf = pyscf.scf.RHF(m)
        with h5py.File(hf_filename, 'r') as f:
            mf.mo_coeff = f['canonical_orbitals'][()]
            mf.mo_energy = f['orbital_energies'][()]
        self.m = m
        self.mf = mf

    def ground_state(self, min_method='slsqp'):
        def energy(params):
            params = qml.numpy.array(params)
            energy = self.circuit(self, params)
            print('energy = ', energy)
            return energy

        def jac(params):
            params = qml.numpy.array(params)
            grad = get_gradient(self.circuit)(self, params)
            return grad
        
        res = minimize(energy, jac=jac, x0=self.theta, method=min_method, tol=1e-12)
        self.theta = res.x

    def hvp(self, v, h=1e-6):
        def grad(x):
            return get_gradient(self.circuit_exc, argnum=2)(self, self.theta, x)
        hvp = np.zeros_like(v)
        if len(v.shape) == 1:
            hvp = (grad(h*v) - grad(-h*v)) / (2*h)
        elif len(v.shape) == 2:
            for i in range(v.shape[1]):
                hvp[:, i] =  (grad(h*v[:,i]) - grad(-h*v[:,i])) / (2*h)
        else:
            raise ValueError
        return hvp

    def hvp_triplet(self, v, h=1e-6):
        def grad(x):
            return get_gradient(self.circuit_exc, argnum=2)(self, self.theta, x, triplet=True)
        hvp = np.zeros_like(v)
        if len(v.shape) == 1:
            hvp = (grad(h*v) - grad(-h*v)) / (2*h)
        elif len(v.shape) == 2:
            for i in range(v.shape[1]):
                hvp[:, i] =  (grad(h*v[:,i]) - grad(-h*v[:,i])) / (2*h)
        else:
            raise ValueError
        return hvp

    def hess_diag_approximate(self, triplet=False):
        if not hasattr(self,"hdiag"):
            orbital_energies = self.mf.mo_energy
            e = np.repeat(orbital_energies, 2) # alpha,beta,alpha,beta
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
            self.hdiag = hdiag
            return hdiag
        else:
            return self.hdiag

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
            for p in range(self.m.nao):
                for q in range(self.m.nao):
                    operator += component[p,q] * qml.FermiC(2*p) * qml.FermiA(2*q)
                    operator += component[p,q] * qml.FermiC(2*p+1) * qml.FermiA(2*q+1)
            operator = qml.jordan_wigner(operator)

            expectation_values.append(self.circuit_operator(self, self.theta, operator))
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
            for p in range(self.m.nao):
                for q in range(self.m.nao):
                    operator += component[p,q] * qml.FermiC(2*p) * qml.FermiA(2*q)
                    operator += sign * component[p,q] * qml.FermiC(2*p+1) * qml.FermiA(2*q+1)
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

    def get_response_fct(self, prop1, prop2, num_imag, omega, history,diag=True):
        """
        Obtain response function by parsing two integrals
        """

        if diag == False:
            raise NameError("Only diagonal response function is implemented")

        hdiag = self.hess_diag_approximate()
        diag_resp = []
        for p1, p2 in zip(prop1, prop2):
            # Calculate linear response vector of prop1
            resp_plus, history = solvers.davidson_response(self.hvp, p1, hdiag, omega=omega, history=history, verbose=True)
            resp_minus, history = solvers.davidson_response(self.hvp, -p1, hdiag, omega=-omega, history=history, verbose=True)
            resp = resp_plus -(-1)**num_imag * resp_minus 
            resp_fct = -(-1)**num_imag * 0.5*np.dot(resp,p2)
            diag_resp.append(resp_fct.real)
            print(omega, resp_fct, f'dim_V={history["V"].shape}', flush=True)

        return diag_resp
    
    def get_OR(self, omega, history,verbose=False):
        """
        Obtain optical rotation in various gauges
        """

        #Get property gradients
        dips_x= self.property_gradient('int1e_r') # V_\mu
        dips_p= self.property_gradient('int1e_ipovlp', approach='statevector') # V_p
        mags= self.property_gradient('int1e_cg_irxp', approach='statevector')  # V_m

        # Length Gauge: <<\mu,m>>
        mu_m = self.get_response_fct(dips_x,mags,num_imag=1,omega=omega,history=history)

        # Velocity Gauge: <<p,m>>
        p_m = self.get_response_fct(dips_p,mags,num_imag=2,omega=omega,history=history)

        # Modified velocity gauge, term 2: <<p,m>>_0
        p_m_0 = self.get_response_fct(dips_p,mags,num_imag=2,omega=0,history=history)

        if verbose:
            print("Length gauge")
            print("xx:", mu_m[0])
            print("yy:", mu_m[1])
            print("zz:", mu_m[2])

            print("Velocity gauge:")
            print("xx:", p_m[0])
            print("yy:", p_m[1])
            print("zz:", p_m[2])

            print("Modified velocity gauge, term 2:")
            print("xx:", p_m_0[0])
            print("yy:", p_m_0[1])
            print("zz:", p_m_0[2])

        # Calculate G' 
        G_length   = - np.sum(mu_m) 
        G_velocity = - np.sum(p_m) / (omega)
        G_modveloc = G_velocity + np.sum(p_m_0) / (omega)

        if verbose:
            print("G' length: ", G_length)
            print("G' velocity: ", G_velocity)
            print("G' modified velocity: ", G_modveloc)

        return G_length, G_velocity, G_modveloc