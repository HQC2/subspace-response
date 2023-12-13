#!/usr/bin/env python

import uccsd
import pennylane as qml
import numpy as np
from pennylane._grad import grad as get_gradient
from scipy.optimize import minimize
from uccsd import UCCSD
import solvers
import excitations

symbols = ['H', 'H', 'H', 'H']
geometry = qml.numpy.array([
[0.0,  0.0,  0.0],
[0.0,  0.0,  2.0],
[1.5,  0.0,  0.0],
[1.5,  0.0,  2.0],
]
                 , requires_grad=False)

symbols = ['Li', 'H']
geometry = qml.numpy.array([
[0.0,  0.0         ,  0.],
[0., 0., 1.6717072740],
                 ], requires_grad=False)*1.8897259886 * 2
#symbols = ['O', 'H', 'H']
#geometry = qml.numpy.array([
#[0.0,  0.0         ,  0.1035174918],
#[0.0,  0.7955612117, -0.4640237459],
#[0.0, -0.7955612117, -0.4640237459],
#], requires_grad=False) * 1.8897259886

basis = 'STO-3G'
charge = 0

class adaptwfn(uccsd.uccsd):
    def __init__(self, symbols, geometry, charge, basis):
        super().__init__(symbols, geometry, charge, basis)

    def ground_state(self, adapt_tol=1e-5):
        # 1) define operator pool
        pool = excitations.spin_adapted_excitations(self.electrons, self.qubits, generalized=True)
        self.excitations_ground_state = []
        self.theta = []
        
        grad_norm = adapt_tol + 1
        adapt_iter = 1
        while grad_norm > adapt_tol:
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
            grad_norm = np.linalg.norm(gradient)
            gradient_pool = gradient[len(previous_excitations_ground_state):]
            add_idx = np.argmax(np.abs(gradient_pool))

            # 3) grow circuit
            self.excitations_ground_state = previous_excitations_ground_state + [pool[add_idx]]
            self.theta = qml.numpy.array(list(self.theta) + [0.])
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
            self.theta = res.x
            adapt_iter += 1


wfn = adaptwfn(symbols, geometry, charge, basis)
#wfn = uccsd.uccsd(symbols, geometry, charge, basis)
wfn.ground_state()

hdiag = wfn.hess_diag_approximate()

dipx, dipy, dipz = wfn.property_gradient('int1e_r')
# davidson_response returns (respose, history)
resp_x, resp_y, resp_z = [solvers.davidson_response(wfn.hvp, dip, hdiag)[0] for dip in [dipx, dipy, dipz]]
print('alpha_xx', np.dot(dipx, resp_x).real)
print('alpha_yy', np.dot(dipy, resp_y).real)
print('alpha_zz', np.dot(dipz, resp_z).real)
