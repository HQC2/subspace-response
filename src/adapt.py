#!/usr/bin/env python

import uccsd
import pennylane as qml
import numpy as np
from pennylane._grad import grad as get_gradient
from scipy.optimize import minimize
from uccsd import UCCSD
import excitations

class adaptwfn(uccsd.uccsd):
    def __init__(self, symbols, geometry, charge, basis, **kwargs):
        super().__init__(symbols, geometry, charge, basis, **kwargs)

    def ground_state(self, adapt_tol=1e-3):
        # 1) define operator pool
        pool = excitations.spin_adapted_excitations(self.electrons, self.qubits, generalized=True)
        self.excitations_ground_state = []
        self.theta = []
        
        max_grad_norm = adapt_tol + 1
        adapt_iter = 1
        last_energy = 0. 
        while max_grad_norm > adapt_tol:
            print(f'Adapt iteration: {adapt_iter}')
            # 2) get gradients of all operators in pool, terminate if below grad_tol
            previous_excitations_ground_state = self.excitations_ground_state
            self.excitations_ground_state = self.excitations_ground_state + pool
            params = qml.numpy.array(list(self.theta) + [0.]*len(pool))

            @qml.qnode(self.device, diff_method="adjoint")
            def circuit(self, params_ground_state):
                UCCSD(params_ground_state, range(self.qubits), self.excitations_ground_state, self.hf_state)
                return qml.expval(self.H)
            self.circuit = circuit

            gradient = get_gradient(self.circuit, argnum=1)(self, params)
            max_grad_norm = np.max(np.abs((gradient)))
            gradient_pool = gradient[len(previous_excitations_ground_state):]
#            add_idx = np.argmax(np.abs(gradient_pool))
            add_indices = np.where(np.abs(gradient_pool) > adapt_tol)[0]

            # 3) grow circuit
            self.excitations_ground_state = previous_excitations_ground_state + [pool[add_idx] for add_idx in add_indices]
            self.theta = qml.numpy.array(list(self.theta) + [0.]*len(add_indices))
            self.num_params = len(self.theta)

            @qml.qnode(self.device, diff_method="adjoint")
            def circuit(self, params_ground_state):
                UCCSD(params_ground_state, range(self.qubits), self.excitations_ground_state, self.hf_state)
                return qml.expval(self.H)
            self.circuit = circuit

            # 4) optimize parameters
            def energy(params):
                params = qml.numpy.array(params)
                energy = self.circuit(self, params)
                print('energy = ', energy, flush=True)
                return energy

            def jac(params):
                params = qml.numpy.array(params)
                grad = get_gradient(self.circuit)(self, params)
                return grad
            
            res = minimize(energy, jac=jac, x0=self.theta, method='slsqp', tol=1e-12)
            
            # prune zero parameters
            zero_theta = np.abs(res.x) < 1e-9
            prune_idx = np.where(zero_theta)[0]
            for index in sorted(prune_idx, reverse=True):
                self.excitations_ground_state.pop(index)
            self.theta = res.x[~zero_theta]

            adapt_iter += 1
            print(f'DeltaE= {res.fun - last_energy:.6e} Added: {add_indices} Removed {prune_idx}')
            last_energy = res.fun
