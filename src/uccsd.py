#!/usr/bin/env python

# based on https://docs.pennylane.ai/en/stable/code/api/pennylane.UCCSD.html

import pennylane as qml
import numpy as np
from pennylane._grad import grad as get_gradient
from scipy.optimize import minimize
from uccsd_circuits import UCCSD, UCCSD_exc, UCCSD_stateprep, UCCSD_iH_exc
import pyscf
import excitations
import copy
from molecular_hamiltonian import get_molecular_hamiltonian
import scipy
import functools


class uccsd(object):

    def __init__(self, symbols, geometry, charge, basis, active_electrons=None, active_orbitals=None):
        atom_str = ''
        for symbol, coord in zip(symbols, geometry):
            atom_str += f'{symbol} {coord[0]} {coord[1]} {coord[2]}; '
        m = pyscf.M(atom=atom_str, basis=basis, charge=charge, unit='bohr')
        mf = pyscf.scf.RHF(m).run()
        self.m = m
        self.mf = mf
        electrons = sum(m.nelec)
        orbitals = m.nao
        active_electrons = active_electrons if active_electrons else electrons
        active_orbitals = active_orbitals if active_orbitals else orbitals
        self.inactive_electrons = electrons - active_electrons
        self.active_electrons = active_electrons
        self.active_orbitals = active_orbitals
        H, qubits = get_molecular_hamiltonian(self, active_electrons=active_electrons, active_orbitals=active_orbitals)
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
            return qml.expval(operator)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit_exc_operator(self, params_ground_state, params_excitation, operator, triplet=False):
            if triplet:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state, excitations_triplet=self.excitations_triplet)
            else:
                UCCSD_exc(params_ground_state, params_excitation, range(self.qubits), self.excitations_ground_state, self.hf_state, excitations_singlet=self.excitations_singlet)
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
        self.circuit_iH_exc = circuit_iH_exc
        self.circuit_iH_exc_operator = circuit_iH_exc_operator
        self.circuit_stateprep = circuit_stateprep
        self.circuit_operator_stateprep = circuit_operator_stateprep
        self.circuit_exc_operator = circuit_exc_operator
        self.circuit_state = circuit_state


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

    def hvp(self, v, h=1e-6, scheme='central', triplet=False):
        grad = lambda x: get_gradient(self.circuit_exc, argnum=2)(self, self.theta, x, triplet=triplet)
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
        return 2.0*hvp

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

    def e3vwp(self, v, w, h=1e-3, triplet=False, scheme='valeev-1'):
        fd_scheme = {
            'central': lambda g, h, v, w: (g(h*v+h*w) - g(-h*v+h*w) - g (h*v-h*w) + g(-h*v-h*w))/(4*h**2),
            'valeev-1': lambda g, h, v, w: (16*(g(-h*w)+g(+h*w)+g(-h*v)+g(+h*v)) 
                                             - (g(2*h*v)+g(-2*h*v)+g(2*h*w)+g(-2*h*w))
                                          - 16*(g(-h*v+h*w)+g(h*v-h*w))
                                             + (g(-2*h*v+2*h*w)+g(2*h*v-2*h*w))
                                          - 30*(g(0*v+0*w))
                                            )/(24*h**2)
                }
        def grad(x):
            return get_gradient(self.circuit_exc, argnum=2)(self, self.theta, x, triplet=triplet)
        assert len(v.shape) == 1
        assert len(w.shape) == 1
        return fd_scheme[scheme](grad, h, v, w)

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

    def V2vp(self, integral, parameter_excitation, triplet=False, h=1e-6, scheme='5-point', optype='fermionic'):
        if isinstance(integral, str):
            ao_integrals = self.m.intor(integral)
        elif isinstance(integral, np.ndarray):
            ao_integrals = integral
        else:
            raise ValueError('Integral should be provided either as a string to evaluate with pyscf m.intor or as a plain numpy array (AO basis).')
        ao_integrals = ao_integrals.reshape(-1, self.m.nao, self.m.nao)
        mo_integrals = np.einsum('uj,xuv,vi->xij', self.mf.mo_coeff, ao_integrals, self.mf.mo_coeff)

        fd_scheme = {
            'forward': lambda g, h, v: g(h*v)/h,
            'central': lambda g, h, v: (g(h*v) - g(-h*v))/(2*h),
            '5-point': lambda g, h, v: (g(-2*h*v) - 8*g(-1*h*v) + 8*g(h*v) - g(2*h*v))/(12*h),
            '7-point': lambda g, h, v: (-g(-3*h*v) + 9*g(-2*h*v) - 45*g(-1*h*v) + 45*g(h*v) - 9*g(2*h*v) + g(3*h*v))/(60*h),
            '9-point': lambda g, h, v: (3*g(-4*h*v) - 32*g(-3*h*v) + 168*g(-2*h*v) - 672*g(-1*h*v) + 672*g(h*v) - 168*g(2*h*v) + 32*g(3*h*v) - 3*g(4*h*v))/(840*h),
            '11-point': lambda g, h, v: (-2*g(-5*h*v) + 25*g(-4*h*v) - 150*g(-3*h*v) + 600*g(-2*h*v) - 2100*g(-1*h*v) + 2100*g(h*v) - 600*g(2*h*v) + 150*g(3*h*v) - 25*g(4*h*v) + 2*g(5*h*v))/(2520*h),
        }
        operator_gradients = []

        for component in mo_integrals:
            # skip null operator
            if np.allclose(component, 0.0):
                operator_gradients.append(np.zeros_like(parameter_excitation))
                continue

            operator = 0.0
            sign = -1 if triplet else 1
            I = self.inactive_electrons//2
            for p in range(self.qubits//2):
                for q in range(self.qubits//2):
                    operator += component[I + p, I + q]*qml.FermiC(2*p)*qml.FermiA(2*q)
                    operator += sign*component[I + p, I + q]*qml.FermiC(2*p + 1)*qml.FermiA(2*q + 1)
            operator = qml.jordan_wigner(operator)
            if optype == 'fermionic':
                circuit = self.circuit_exc_operator
            elif optype == 'iH':
                circuit = self.circuit_iH_exc_operator
            else:
                raise ValueError
            grad = lambda v: get_gradient(circuit, argnum=2)(self, self.theta, v, operator, triplet=triplet)
            operator_gradient = fd_scheme[scheme](grad, h, parameter_excitation)
            operator_gradients.append(operator_gradient)
        return np.array(operator_gradients)

    def apply_tensor_op(self, op, basis_state):
        statevector = np.zeros(2**len(basis_state), dtype=np.complex128)
        phase_map = {
                ('X', 0): 1,
                ('X', 1): 1,
                ('Y', 0): 1j,
                ('Y', 1): -1j,
                ('Z', 0): 1,
                ('Z', 1): -1,
                }
        flip_map = {
                ('X', 0): 1,
                ('X', 1): 1,
                ('Y', 0): 1,
                ('Y', 1): 1,
                ('Z', 0): 0,
                ('Z', 1): 0,
                }
        flip = np.zeros_like(basis_state)
        for weight, pauli in zip(*op.terms()):
            if pauli.label() == 'I':
                index = np.sum((basis_state)*2**(np.arange(self.qubits)[::-1]))
                statevector[index] += weight
                continue
            # Z accumulates (-1) phase on |1> qubits
            # X,Y flips bit
            # Y accumulates (i/-i) phase on (|0>/|1>) 
            phase = 1
            flip[:] = 0
            if pauli.arithmetic_depth > 0:
                operands = pauli.operands
            else:
                operands = [pauli]
            for operand in operands:
                qubit = operand.wires[0]
                qubit_state = basis_state[qubit]
                label = operand.label()
                flip[qubit] = flip_map[(label, qubit_state)]
                phase *= phase_map[(label, qubit_state)]
            index = np.sum((basis_state ^ flip)*2**(np.arange(self.qubits)[::-1]))
            statevector[index] += weight*phase
        return statevector

    def V2_contraction(self, integral, I_vector, I_vector_dag, J_vector, J_vector_dag, triplet=False):
        if isinstance(integral, str):
            ao_integrals = self.m.intor(integral)
        elif isinstance(integral, np.ndarray):
            ao_integrals = integral
        else:
            raise ValueError('Integral should be provided either as a string to evaluate with pyscf m.intor or as a plain numpy array (AO basis).')
        ao_integrals = ao_integrals.reshape(-1, self.m.nao, self.m.nao)
        mo_integrals = np.einsum('uj,xuv,vi->xij', self.mf.mo_coeff, ao_integrals, self.mf.mo_coeff)

        fd_scheme = {
            'forward': lambda g, h, v: g(h*v)/h,
            'central': lambda g, h, v: (g(h*v) - g(-h*v))/(2*h),
            '5-point': lambda g, h, v: (g(-2*h*v) - 8*g(-1*h*v) + 8*g(h*v) - g(2*h*v))/(12*h),
            '7-point': lambda g, h, v: (-g(-3*h*v) + 9*g(-2*h*v) - 45*g(-1*h*v) + 45*g(h*v) - 9*g(2*h*v) + g(3*h*v))/(60*h),
            '9-point': lambda g, h, v: (3*g(-4*h*v) - 32*g(-3*h*v) + 168*g(-2*h*v) - 672*g(-1*h*v) + 672*g(h*v) - 168*g(2*h*v) + 32*g(3*h*v) - 3*g(4*h*v))/(840*h),
            '11-point': lambda g, h, v: (-2*g(-5*h*v) + 25*g(-4*h*v) - 150*g(-3*h*v) + 600*g(-2*h*v) - 2100*g(-1*h*v) + 2100*g(h*v) - 600*g(2*h*v) + 150*g(3*h*v) - 25*g(4*h*v) + 2*g(5*h*v))/(2520*h),
        }
        operator_gradients = []

        for component in mo_integrals:
            # skip null operator
            if np.allclose(component, 0.0):
                operator_gradients.append(np.zeros_like(parameter_excitation))
                continue

            operator = 0.0
            sign = -1 if triplet else 1
            I = self.inactive_electrons//2
            for p in range(self.qubits//2):
                for q in range(self.qubits//2):
                    operator += component[I + p, I + q]*qml.FermiC(2*p)*qml.FermiA(2*q)
                    operator += sign*component[I + p, I + q]*qml.FermiC(2*p + 1)*qml.FermiA(2*q + 1)
            operator = qml.jordan_wigner(operator)

            excitation_operators = []
            def mEpq(p,q):
                return qml.jordan_wigner(qml.FermiC(2*p)*qml.FermiA(2*q) + qml.FermiC(2*p+1)*qml.FermiA(2*q+1))
            def T1(i,a):
                return mEpq(a,i)/np.sqrt(2)
            def T2t(i,j,a,b):
                return (mEpq(a,i)@mEpq(b,j)-mEpq(a,j)@mEpq(b,i))/(2*np.sqrt(3))
            def T2(i,j,a,b):
                if (i==j) and (a==b):
                    prefactor = 0.25
                elif (i==j) or (a==b):
                    prefactor = 0.5 / np.sqrt(2)
                else:
                    prefactor = 0.5
                return prefactor*(mEpq(a,i)@mEpq(b,j)+mEpq(a,j)@mEpq(b,i))
            for e,w in self.excitations_singlet:
                exci = [p // 2 for p in e[0]]
                if len(exci) == 2:
                    i, a = exci
                    excitation_operators.append(T1(i,a))
                elif len(exci) == 4:
                    i,j,a,b = exci
                    if len(w)==6:
                        excitation_operators.append(T2t(i,j,a,b))
                    else:
                        excitation_operators.append(T2(i,j,a,b))

            op_I = sum([I_vector[i] * excitation_operators[i] for i in range(len(excitation_operators))]).simplify()
            op_I_dag = sum([I_vector_dag[i] * excitation_operators[i] for i in range(len(excitation_operators))]).simplify()
            op_J = sum([J_vector[i] * excitation_operators[i] for i in range(len(excitation_operators))]).simplify()
            op_J_dag = sum([J_vector_dag[i] * excitation_operators[i] for i in range(len(excitation_operators))]).simplify()

            # (dagger,dagger) term
            # -<Psi|I'J'O|Psi>
            # apply I'J' to <Psi|
            total = 0.0
            op_IJ = (op_J_dag @ op_I_dag).simplify()
            IJ_statevec = self.apply_tensor_op(op_IJ, self.hf_state)
            hf_statevec = np.zeros(2**self.qubits)
            index = np.sum(2**(np.arange(self.qubits)[::-1])*self.hf_state)
            hf_statevec[index] = 1
            plus_statevec = IJ_statevec + hf_statevec
            IJ_norm  = np.linalg.norm(IJ_statevec)
            plus_norm  = np.linalg.norm(plus_statevec)

            X_plus = self.circuit_operator_stateprep(self, self.theta, plus_statevec/plus_norm, operator)
            X_IJ = self.circuit_operator_stateprep(self, self.theta, IJ_statevec/IJ_norm, operator)
            X_0 = self.circuit_operator_stateprep(self, self.theta, hf_statevec, operator)
            term = -(0.5*X_plus*plus_norm**2 - 0.5*X_IJ*IJ_norm**2- 0.5*X_0)
            print(X_plus,plus_norm, X_IJ,IJ_norm, X_0)
            print(term)
            total += term
            
            # (dagger,.) term
            # <Psi|I'OJ - I'JO|Psi>
            # I'JO:
            op_IJ = (op_J.adjoint() @ op_I_dag).simplify()
            IJ_statevec = self.apply_tensor_op(op_IJ, self.hf_state)
            hf_statevec = np.zeros(2**self.qubits)
            index = np.sum(2**(np.arange(self.qubits)[::-1])*self.hf_state)
            hf_statevec[index] = 1
            plus_statevec = IJ_statevec + hf_statevec
            IJ_norm  = np.linalg.norm(IJ_statevec)
            plus_norm  = np.linalg.norm(plus_statevec)

            X_plus = self.circuit_operator_stateprep(self, self.theta, plus_statevec/plus_norm, operator)
            X_IJ = self.circuit_operator_stateprep(self, self.theta, IJ_statevec/IJ_norm, operator)
            X_0 = self.circuit_operator_stateprep(self, self.theta, hf_statevec, operator)
            term = -(0.5*X_plus*plus_norm**2 - 0.5*X_IJ*IJ_norm**2- 0.5*X_0)
            print(term)
            total += term
            # I'OJ:
            I_statevec = self.apply_tensor_op(op_I_dag, self.hf_state)
            J_statevec = self.apply_tensor_op(op_J, self.hf_state)
            plus_statevec = I_statevec + J_statevec
            I_norm  = np.linalg.norm(I_statevec)
            J_norm  = np.linalg.norm(J_statevec)
            plus_norm  = np.linalg.norm(plus_statevec)
            
            if plus_norm > 1e-12:
                X_plus = self.circuit_operator_stateprep(self, self.theta, plus_statevec/plus_norm, operator)
            else:
                X_plus = 0.
            X_I = self.circuit_operator_stateprep(self, self.theta, I_statevec/I_norm, operator)
            X_J = self.circuit_operator_stateprep(self, self.theta, J_statevec/J_norm, operator)
            term = -(0.5*X_plus*plus_norm**2 - 0.5*X_I*I_norm**2- 0.5*X_J*J_norm**2)
            print(term)
            total -= term

            # (.,dagger) term
            # <Psi|J'OI - OJ'I|Psi>
            # OJ'I:
            op_IJ = (op_J_dag.adjoint() @ op_I).simplify()
            IJ_statevec = self.apply_tensor_op(op_IJ, self.hf_state)
            hf_statevec = np.zeros(2**self.qubits)
            index = np.sum(2**(np.arange(self.qubits)[::-1])*self.hf_state)
            hf_statevec[index] = 1
            plus_statevec = IJ_statevec + hf_statevec
            IJ_norm  = np.linalg.norm(IJ_statevec)
            plus_norm  = np.linalg.norm(plus_statevec)

            if plus_norm > 1e-12:
                X_plus = self.circuit_operator_stateprep(self, self.theta, plus_statevec/plus_norm, operator)
            else:
                X_plus = 0.
            X_IJ = self.circuit_operator_stateprep(self, self.theta, IJ_statevec/IJ_norm, operator)
            X_0 = self.circuit_operator_stateprep(self, self.theta, hf_statevec, operator)
            term = -(0.5*X_plus*plus_norm**2 - 0.5*X_IJ*IJ_norm**2- 0.5*X_0)
            print(term)
            total += term
            # J'OI:
            I_statevec = self.apply_tensor_op(op_I, self.hf_state)
            J_statevec = self.apply_tensor_op(op_J_dag, self.hf_state)
            plus_statevec = I_statevec + J_statevec
            I_norm  = np.linalg.norm(I_statevec)
            J_norm  = np.linalg.norm(J_statevec)
            plus_norm  = np.linalg.norm(plus_statevec)

            if plus_norm > 1e-12:
                X_plus = self.circuit_operator_stateprep(self, self.theta, plus_statevec/plus_norm, operator)
            else:
                X_plus = 0.
            X_I = self.circuit_operator_stateprep(self, self.theta, I_statevec/I_norm, operator)
            X_J = self.circuit_operator_stateprep(self, self.theta, J_statevec/J_norm, operator)
            term = -(0.5*X_plus*plus_norm**2 - 0.5*X_I*I_norm**2- 0.5*X_J*J_norm**2)
            print(term)
            total -= term

            # (.,.) term
            # -<Psi|OJI|Psi>
            op_IJ = (op_J @ op_I).simplify()
            IJ_statevec = self.apply_tensor_op(op_IJ, self.hf_state)
            hf_statevec = np.zeros(2**self.qubits)
            index = np.sum(2**(np.arange(self.qubits)[::-1])*self.hf_state)
            hf_statevec[index] = 1
            plus_statevec = IJ_statevec + hf_statevec
            IJ_norm  = np.linalg.norm(IJ_statevec)
            plus_norm  = np.linalg.norm(plus_statevec)

            if plus_norm > 1e-12:
                X_plus = self.circuit_operator_stateprep(self, self.theta, plus_statevec/plus_norm, operator)
            else:
                X_plus = 0.
            X_IJ = self.circuit_operator_stateprep(self, self.theta, IJ_statevec/IJ_norm, operator)
            X_0 = self.circuit_operator_stateprep(self, self.theta, hf_statevec, operator)
            term = -(0.5*X_plus*plus_norm**2 - 0.5*X_IJ*IJ_norm**2- 0.5*X_0)
            print(term)
            total += term
            print('Total:', total)

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
        return 0.5*result


