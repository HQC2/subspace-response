"""Adaptive ground-state selection with scalar Trotter blocks.

USCC uses a spin-adapted amplitude-times-integral heuristic. SPQE uses exact
transformed residuals with either VQE energy or PQE residual optimization.
Gradient selection scores appended scalar directions on the current state.
"""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pennylane as qml
import pyscf
from scipy.optimize import minimize

from excitations import Excitation, singlet_excitations, singlet_excitations_for_block
from molecular_hamiltonian import ActiveSpaceIntegrals, get_active_space_integrals
from stateprep import stateprep
from ucc_circuits import apply_ucc
from uscc_optimizers import excitation_curvature, minimize_metric
from spqe import ReferenceProjector, SPQESelection, solve_pqe
from gradient_selection import GradientSelection

FORMAT_VERSION = 1
CONVENTION = 'canonical-particle-ascending-hole-descending;scalar-mgs-v1;exp(theta*(T-Tdagger))'


def block_key(term):
    return tuple(sorted(h//2 for h in term.holes)), tuple(sorted(p//2 for p in term.particles))


def block_id(block):
    return ','.join(map(str, block[0])) + '->' + ','.join(map(str, block[1]))


class USCCSelector:
    """Max-product screening followed by promotion of whole scalar blocks.

    The one-body coefficient is h[a,i]. The two-body coefficient is the
    antisymmetrized coefficient of canonical tau in (1/2) V a†a†aa. Scalar
    amplitudes are expanded into monomial coefficients sum_mu theta_mu*c_mu,m.
    This is a first-order proposal surrogate, not a CI coefficient estimator.
    All selected excitation ranks can supply amplitudes for further proposals;
    max_rank limits the resulting candidates.
    """
    def __init__(self, integrals, max_rank=4, orbital_irreps=None):
        if not isinstance(max_rank, int) or max_rank < 1:
            raise ValueError('max_rank must be a positive integer')
        if orbital_irreps is not None and (len(orbital_irreps) != integrals.qubits//2 or any(not isinstance(i, (int, np.integer)) or i < 0 for i in orbital_irreps)):
            raise ValueError('Expected nonnegative Abelian XOR irrep IDs')
        self.integrals = integrals
        self.max_rank = min(max_rank, integrals.electrons, integrals.qubits-integrals.electrons)
        self.requested_max_rank = max_rank
        self.orbital_irreps = None if orbital_irreps is None else tuple(map(int, orbital_irreps))
        self.channels = integrals.excitation_channels()

    def allowed(self, block):
        if len(block[0]) > self.max_rank:
            return False
        if self.orbital_irreps is not None:
            irrep = 0
            for orbital in block[0]+block[1]:
                irrep ^= self.orbital_irreps[orbital]
            if irrep:
                return False
        return True

    def base_scores(self):
        scores = {}
        for channel in self.channels:
            block = block_key(channel)
            if self.allowed(block):
                score = abs(channel.coefficient)
                if score >= scores.get(block, (-1., ''))[0]:
                    scores[block] = (score, f'h{channel.rank}')
        return scores

    def scores(self, excitations, parameters):
        scores = self.base_scores()
        amplitudes = {}
        for excitation, theta in zip(excitations, parameters):
            for term in excitation.terms:
                key = term.holes, term.particles
                amplitudes[key] = amplitudes.get(key, 0.) + float(theta)*term.coefficient
        nonzero_channels = [c for c in self.channels if abs(c.coefficient) > 0]
        for (holes, particles), amplitude in sorted(amplitudes.items()):
            if abs(amplitude) == 0:
                continue
            for channel in nonzero_channels:
                rank = len(holes)+channel.rank
                if rank > self.max_rank or set(holes) & set(channel.holes) or set(particles) & set(channel.particles):
                    continue
                hs = tuple(sorted(holes+channel.holes))
                ps = tuple(sorted(particles+channel.particles))
                block = tuple(sorted(h//2 for h in hs)), tuple(sorted(p//2 for p in ps))
                if not self.allowed(block):
                    continue
                score = abs(amplitude*channel.coefficient)
                if score > scores.get(block, (-1., ''))[0]:
                    scores[block] = (score, f't{len(holes)}*h{channel.rank}')
        return scores

    def select(self, excitations, parameters, epsilon):
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError('Selection threshold must be finite and nonnegative')
        present = {e.id for e in excitations}
        blocks = {e.block for e in excitations}
        selected = {block for block in blocks if all(g.id in present for g in singlet_excitations_for_block(*block))}
        return [(block, score, reason) for block, (score, reason) in sorted(self.scores(excitations, parameters).items(), key=lambda item: (-item[1][0], item[0])) if block not in selected and score >= epsilon]


class USCC:
    """Reusable rank-independent ground-state wave function and USCC driver.

    Parameters are physical generator coefficients, not legacy gate angles.
    ``apply_ucc`` and the serialized ordered generators remain usable by later
    response code, independently of this ground-state selection loop.
    """
    def __init__(self, integrals, *, metadata=None, max_rank=4, seed='screened', selection='uscc', trotter_steps=1, device='lightning.qubit', orbital_irreps=None, backend='sector', sector_threads=None, sector_kernels='auto', rotation_threads=None):
        if selection not in ('uscc', 'spqe', 'gradient'):
            raise ValueError('selection must be uscc, spqe or gradient')
        self.selection = selection
        if backend not in ('sector','pennylane'):
            raise ValueError('backend must be sector or pennylane')
        if seed not in ('sd', 'screened'):
            raise ValueError('seed must be sd or screened')
        if not isinstance(trotter_steps, int) or trotter_steps < 1:
            raise ValueError('trotter_steps must be a positive integer')
        if sector_threads is not None and (type(sector_threads) is not int or sector_threads < 1):
            raise ValueError('sector_threads must be a positive integer or None')
        self.sector_threads = sector_threads
        if sector_kernels not in ('auto','native','numba','numpy'):
            raise ValueError('sector_kernels must be auto, native, numba or numpy')
        if rotation_threads is not None and (type(rotation_threads) is not int or rotation_threads < 1):
            raise ValueError('rotation_threads must be a positive integer or None')
        self.sector_kernels = sector_kernels
        self.rotation_threads = rotation_threads
        self.integrals = integrals
        # The sector path uses spatial integrals directly. Materialize its JW
        # observable only when a caller actually requests H or a circuit.
        self._H = None
        self._integral_hamiltonian = None
        self.backend = backend
        self._simulator = None
        self.qubits = integrals.qubits
        self.hf_state = np.array([1]*integrals.electrons+[0]*(self.qubits-integrals.electrons), dtype=int)
        self.metadata = {} if metadata is None else metadata
        self.selector = USCCSelector(integrals, max_rank, orbital_irreps)
        self.seed = seed
        self.trotter_steps = trotter_steps
        self.device_name = device
        self.device = qml.device(device, wires=self.qubits)
        self.excitations = []
        self.theta = np.zeros(0)
        self.history = []
        self.provenance = {}
        self.schedule = None
        self.threshold_index = 0
        self.initialized = False
        self.needs_optimization = False
        self.stop_reason = 'not_started'
        self.optimization_settings = dict(gradient_tolerance=1e-6, residual_tolerance=1e-6, energy_tolerance=1e-10, maxiter=300, method='lbfgs')
        self._optimizer_state = None
        self._spqe_selector = None
        self._gradient_selector = None
        self._projective_projector = None

        @qml.qnode(self.device, diff_method='adjoint')
        def expectation_circuit(parameters, operator, vector=None):
            if vector is None:
                qml.BasisState(self.hf_state, wires=range(self.qubits))
            else:
                if vector.shape not in ((2**self.qubits,1), (1,2**self.qubits)):
                    raise ValueError('Prepared state dimension does not match the active register')
                stateprep(vector)
            apply_ucc(parameters, self.excitations, range(self.qubits), self.trotter_steps)
            if isinstance(operator, list):
                return [qml.expval(op) for op in operator]
            return qml.expval(operator)

        @qml.qnode(self.device, diff_method=None)
        def state_circuit(parameters, vector=None):
            if vector is None:
                qml.BasisState(self.hf_state, wires=range(self.qubits))
            else:
                if vector.shape not in ((2**self.qubits,1), (1,2**self.qubits)):
                    raise ValueError('Prepared state dimension does not match the active register')
                stateprep(vector)
            apply_ucc(parameters, self.excitations, range(self.qubits), self.trotter_steps)
            return qml.state()
        self.circuit = expectation_circuit
        self.circuit_state = state_circuit

        @qml.qnode(self.device, diff_method=None)
        def adjoint_state_circuit(parameters, vector):
            qml.StatePrep(vector, wires=range(self.qubits))
            qml.adjoint(apply_ucc)(parameters, self.excitations, range(self.qubits), self.trotter_steps)
            return qml.state()
        self.circuit_adjoint_state = adjoint_state_circuit

    @property
    def H(self):
        if self._H is None:
            self._H = self.integrals.hamiltonian()
            self._integral_hamiltonian = self._H
        return self._H

    @H.setter
    def H(self, operator):
        self._H = operator

    @classmethod
    def from_molecule(cls, symbols, geometry, charge=0, basis='sto-3g', *, unit='bohr', active_electrons=None, active_orbitals=None, **kwargs):
        geometry = np.asarray(geometry, dtype=float)
        if geometry.shape != (len(symbols),3) or not np.all(np.isfinite(geometry)):
            raise ValueError('Expected one finite xyz coordinate per atom')
        if unit.lower() not in ('bohr', 'angstrom'):
            raise ValueError('unit must be bohr or angstrom')
        m = pyscf.M(atom=[(s, tuple(x)) for s,x in zip(symbols,geometry)], charge=charge, basis=basis, unit=unit, spin=0, verbose=0)
        if m.nelectron % 2:
            raise ValueError('The initial implementation requires a closed-shell reference')
        mf = pyscf.scf.RHF(m).run()
        if not mf.converged:
            raise RuntimeError('RHF reference did not converge')
        ne = m.nelectron if active_electrons is None else active_electrons
        no = m.nao if active_orbitals is None else active_orbitals
        if not isinstance(ne, int) or not isinstance(no, int) or ne < 0 or ne % 2 or ne > m.nelectron or no < ne//2 or no+(m.nelectron-ne)//2 > m.nao:
            raise ValueError('Invalid closed-shell frozen-core active space')
        setup = SimpleNamespace(m=m, mf=mf, inactive_orbitals=(m.nelectron-ne)//2, active_orbitals=no, active_electrons=ne)
        integrals = get_active_space_integrals(setup)
        metadata = dict(symbols=list(symbols), geometry=geometry.tolist(), unit=unit.lower(), charge=charge, basis=basis, active_electrons=ne, active_orbitals=no, inactive_orbitals=setup.inactive_orbitals, mo_coeff=mf.mo_coeff.tolist())
        result = cls(integrals, metadata=metadata, **kwargs)
        result.m, result.mf = m, mf
        return result

    def set_excitations(self, excitations, parameters=None):
        """Install an ordered fixed ansatz; useful independently of selection."""
        excitations = list(excitations)
        if len({e.id for e in excitations}) != len(excitations):
            raise ValueError('Duplicate excitation IDs')
        for e in excitations:
            if any(h >= self.integrals.electrons for t in e.terms for h in t.holes) or any(p < self.integrals.electrons or p >= self.qubits for t in e.terms for p in t.particles):
                raise ValueError('Excitation is incompatible with the active reference')
            if not self.selector.allowed(e.block):
                raise ValueError('Excitation violates the requested rank or symmetry')
        parameters = np.zeros(len(excitations)) if parameters is None else np.asarray(parameters, dtype=float)
        if parameters.shape != (len(excitations),) or not np.all(np.isfinite(parameters)):
            raise ValueError('Expected one finite parameter per excitation')
        self.excitations = excitations
        self.theta = parameters.copy()
        self.initialized = True
        self.needs_optimization = True
        self.threshold_index = 0
        self.history = []
        self.provenance = {}
        self.stop_reason = 'not_started'
        self._optimizer_state = None

    def _sector_simulator(self):
        if self._simulator is None:
            from uscc_simulator import SectorSimulator
            self._simulator = SectorSimulator(self.integrals, sector_threads=self.sector_threads, kernels=self.sector_kernels, rotation_threads=self.rotation_threads)
        return self._simulator

    def _sector_state(self, parameters, statevector):
        simulator = self._sector_simulator()
        initial = None if statevector is None else simulator.project_input(statevector)
        if statevector is not None and initial is None:
            return None  # Preserve full-register support for other sectors.
        return simulator.state(parameters,self.excitations,self.trotter_steps,initial)

    def expectation(self, operator=None, parameters=None, statevector=None):
        parameters = self.theta if parameters is None else parameters
        if self.backend == 'sector':
            state = self._sector_state(parameters,statevector)
            if state is not None:
                simulator = self._sector_simulator()
                operator = self._H if operator is None else operator
                norm = np.vdot(state,state).real
                def measure(op):
                    if op is self._integral_hamiltonian:
                        return np.vdot(state,simulator.apply_hamiltonian(state)).real/norm
                    return simulator.expectation(state,op)/norm
                return [measure(op) for op in operator] if isinstance(operator,list) else measure(operator)
        return self.circuit(parameters,self.H if operator is None else operator,statevector)

    def state(self, parameters=None, statevector=None):
        parameters = self.theta if parameters is None else parameters
        if self.backend == 'sector':
            state = self._sector_state(parameters,statevector)
            if state is not None:
                return self._sector_simulator().full_state(state)
        return self.circuit_state(parameters,statevector)

    def energy_and_gradient(self, parameters):
        if self.backend == 'sector':
            hamiltonian = None if self._H is self._integral_hamiltonian else self.H
            return self._sector_simulator().energy_and_gradient(parameters,self.excitations,self.trotter_steps,hamiltonian)
        parameters = qml.numpy.array(parameters, requires_grad=True)
        if not len(parameters):
            return float(self.expectation(parameters=parameters)), np.zeros(0)
        gradient_function = qml.grad(self.circuit, 0)
        gradient = gradient_function(parameters, self.H)
        return float(gradient_function.forward), np.asarray(gradient,dtype=float)

    def diagnostics(self):
        if self.backend == 'sector':
            simulator = self._sector_simulator()
            state = simulator.state(self.theta,self.excitations,self.trotter_steps)
            return dict(particle_number=float(self.integrals.electrons*np.vdot(state,state).real), spin_z=0., spin_squared=simulator.spin_squared(state))
        operators = [qml.qchem.particle_number(self.qubits), qml.qchem.spinz(self.qubits), qml.qchem.spin2(self.integrals.electrons, self.qubits)]
        number, sz, s2 = self.expectation(operators)
        return dict(particle_number=float(number), spin_z=float(sz), spin_squared=float(s2))

    def projective_residual_state(self, parameters=None):
        """Return E, U†(H-E)U|HF>, and its computational basis indices.

        The sector backend never expands the full register or transformed H.
        The PennyLane path is a full-register oracle with the same circuit.
        """
        parameters = self.theta if parameters is None else np.asarray(parameters, dtype=float)
        if self.backend == 'sector':
            simulator = self._sector_simulator()
            state = simulator.state(parameters, self.excitations, self.trotter_steps)
            applied = (simulator.apply_hamiltonian(state) if self._H is self._integral_hamiltonian
                       else simulator.observable_matrix(self.H) @ state)
            energy = float(np.vdot(state, applied).real)
            residual = simulator.apply_unitary(applied-energy*state, parameters, self.excitations,
                                               self.trotter_steps, adjoint=True)
            return energy, residual, simulator.indices
        state = np.asarray(self.circuit_state(parameters))
        try:
            hamiltonian = self.H.sparse_matrix(wire_order=range(self.qubits))
        except qml.operation.SparseMatrixUndefinedError:
            hamiltonian = qml.matrix(self.H, wire_order=range(self.qubits))
        applied = hamiltonian @ state
        energy = float(np.vdot(state, applied).real)
        vector = applied-energy*state
        norm = np.linalg.norm(vector)
        residual = np.asarray(self.circuit_adjoint_state(parameters, vector/norm))*norm if norm else vector
        return energy, residual, np.arange(len(state))

    def projective_residuals(self, parameters=None):
        """Energy and residuals projected onto the active scalar reference CSFs."""
        energy, vector, indices = self.projective_residual_state(parameters)
        return energy, self._reference_projector(indices).project(vector)

    def _projection_context(self, indices):
        hf_index = ((1 << self.integrals.electrons)-1) << (self.qubits-self.integrals.electrons)
        simulator = self._sector_simulator() if self.backend == 'sector' else None
        address = simulator.address if simulator is not None else range(len(indices))
        return hf_index, address, (self.backend, id(simulator), self.qubits, hf_index)

    def _reference_projector(self, indices):
        hf_index, address, context = self._projection_context(indices)
        ids = tuple(e.id for e in self.excitations)
        cached = self._projective_projector
        if cached is None or cached[0] != context or cached[1].ids != ids:
            cached = self._projective_projector = (context, ReferenceProjector(
                self.excitations, address, hf_index, self.qubits, len(indices)))
        return cached[1]

    def _select(self, epsilon):
        if self.selection == 'uscc':
            return self.selector.select(self.excitations, self.theta, epsilon), {}
        if self.selection == 'gradient':
            simulator = self._sector_simulator()
            context = (id(simulator), id(self.selector), self.selector.max_rank, self.selector.orbital_irreps)
            cached = self._gradient_selector
            if cached is None or cached[0] != context:
                cached = self._gradient_selector = (context, GradientSelection(
                    simulator, self.selector))
            if self.backend == 'sector':
                state = simulator.state(self.theta, self.excitations, self.trotter_steps)
                if self._H is self._integral_hamiltonian:
                    applied = simulator.apply_hamiltonian(state)
                else:
                    matrix = simulator.observable_matrix(self.H)
                    applied = matrix @ state
            else:
                # Prepare and measure with the full-register reference backend;
                # candidate rotations conserve the same fixed N_alpha/N_beta.
                full_state = np.asarray(self.circuit_state(self.theta))
                try:
                    matrix = self.H.sparse_matrix(wire_order=range(self.qubits))
                except qml.operation.SparseMatrixUndefinedError:
                    matrix = qml.matrix(self.H, wire_order=range(self.qubits))
                state = full_state[simulator.indices]
                applied = (matrix @ full_state)[simulator.indices]
            return cached[1].select(state, applied, self.excitations, epsilon)
        _, vector, indices = self.projective_residual_state()
        hf_index, address, context = self._projection_context(indices)
        context += (id(self.selector), self.selector.max_rank, self.selector.orbital_irreps)
        cached = self._spqe_selector
        if cached is None or cached[0] != context:
            cached = self._spqe_selector = (context, SPQESelection(
                indices, hf_index, self.qubits, self.selector, address))
        return cached[1].select(vector, self.excitations, epsilon)

    def resources(self):
        # A rank-r pure excitation has 2**(2*r-1) JW Pauli words. Every word
        # has X/Y at the endpoints and the same Z support between alternating
        # sorted endpoint pairs. Count that support without expanding words.
        rotations, cnots = 0, 0
        for excitation in self.excitations:
            for term in excitation.terms:
                modes = sorted(term.holes + term.particles)
                count = 1 << (2*term.rank-1)
                support = sum(b-a+1 for a,b in zip(modes[::2],modes[1::2]))
                rotations += count
                cnots += count*2*(support-1)
        return dict(parameters=len(self.theta), ranks={str(r):sum(e.rank == r for e in self.excitations) for r in range(1,self.selector.requested_max_rank+1)}, pauli_rotations=rotations*self.trotter_steps, estimated_parity_cnots=cnots*self.trotter_steps)

    def _append_blocks(self, proposals, max_operators=None):
        added, limited = [], False
        present = {e.id for e in self.excitations}
        for block, score, reason in proposals:
            cached = (self._spqe_selector if self.selection == 'spqe'
                      else self._gradient_selector if self.selection == 'gradient' else None)
            try:
                pool = cached[1].generators(block) if cached is not None else None
            except KeyError:
                pool = None  # Explicit seeds or manually supplied proposals.
            if pool is None:
                pool = singlet_excitations_for_block(*block)
            generators = tuple(e for e in pool if e.id not in present)
            if max_operators is not None and len(self.excitations)+len(added)+len(generators) > max_operators:
                limited = True
                break
            self.provenance[block_id(block)] = dict(score=float(score), source=reason)
            added.extend(generators)
            present.update(e.id for e in generators)
        if added:
            if self.selection == 'spqe':
                # Circuit order is the reverse of SPQE's algebraic product.
                self.excitations[0:0] = reversed(added)
                self.theta = np.pad(self.theta, (len(added),0))
                self._optimizer_state = None
            else:
                self.excitations.extend(added)
                self.theta = np.pad(self.theta, (0,len(added)))
        self.needs_optimization = bool(added) or self.needs_optimization
        return [e.id for e in added], limited

    def _optimizer_context(self):
        # Custom observables can be replaced independently of the integrals.
        # Their runtime identity is valid in memory, but is never checkpointed.
        observable = None if self._H is self._integral_hamiltonian else id(self._H)
        digest = sha256()
        for array in (self.integrals.one_body,self.integrals.two_body):
            digest.update(np.asarray(array,dtype='<f8').tobytes())
        digest.update(json.dumps(dict(constant=self.integrals.constant,electrons=self.integrals.electrons),sort_keys=True).encode())
        return dict(fingerprint=digest.hexdigest(), trotter_steps=self.trotter_steps, observable=observable)

    def _curvature_history(self, *, strict=False):
        state = self._optimizer_state
        if state is None:
            return None
        try:
            n = len(state['ids'])
            anchor = np.asarray(state['theta'], dtype=float)
            s = np.asarray(state['s'], dtype=float)
            y = np.asarray(state['y'], dtype=float)
            if state['version'] != 1 or anchor.shape != (n,) or s.ndim != 2 or s.shape != y.shape or s.shape[1] != n or len(s)>10:
                raise ValueError('Invalid optimizer curvature state dimensions')
            with np.errstate(over='ignore',invalid='ignore'):
                sy = np.sum(s*y,axis=1)
            if not all(np.all(np.isfinite(a)) for a in (anchor,s,y,sy)) or np.any(sy <= 0):
                raise ValueError('Invalid optimizer curvature state values')
            if n>len(self.theta) or state['ids'] != [e.id for e in self.excitations[:n]] or state['context'] != self._optimizer_context() or not np.array_equal(anchor,self.theta[:n]) or np.any(self.theta[n:]):
                if strict:
                    raise ValueError('Optimizer curvature state does not match the saved wave function')
                return None
            return [(np.pad(a,(0,len(self.theta)-n)),np.pad(b,(0,len(self.theta)-n))) for a,b in zip(s,y)]
        except (KeyError, TypeError, ValueError) as error:
            if strict:
                raise ValueError('Invalid optimizer curvature state') from error
            return None

    def optimize(self, *, gradient_tolerance=1e-6, residual_tolerance=1e-6, energy_tolerance=1e-10, maxiter=300, method='lbfgs'):
        """Optimize with a physical diagonal metric and reusable L-BFGS pairs.

        L-BFGS and DIIS require the requested L2 gradient tolerance; a tiny
        energy change alone is insufficient. Failed calls preserve theta and
        the previously accepted curvature state. DIIS starts with two metric
        gradient steps; SLSQP and dense BFGS retain their SciPy implementations.
        PQE instead solves projected equations to residual_tolerance, using
        gap-preconditioned updates and DIIS without energy gradients.
        """
        if not np.isfinite(gradient_tolerance) or not np.isfinite(residual_tolerance) or not np.isfinite(energy_tolerance) or gradient_tolerance <= 0 or residual_tolerance <= 0 or energy_tolerance <= 0 or type(maxiter) is not int or maxiter < 1:
            raise ValueError('Optimization tolerances and iteration count must be positive')
        method = method.lower()
        if method not in ('slsqp','bfgs','lbfgs','diis','pqe'):
            raise ValueError('Optimizer must be slsqp, bfgs, lbfgs, diis or pqe')
        if method == 'pqe':
            return self._optimize_pqe(residual_tolerance=residual_tolerance, maxiter=maxiter)
        initial = self.theta.copy()
        old_energy, old_gradient = self.energy_and_gradient(initial)
        last = [initial.copy(), (old_energy,old_gradient)]
        evaluations = 1
        def objective(parameters):
            nonlocal evaluations
            if not np.array_equal(parameters,last[0]):
                last[:] = [np.array(parameters,copy=True),self.energy_and_gradient(parameters)]
                evaluations += 1
            return last[1]
        reused = 0
        if not len(initial) or np.linalg.norm(old_gradient) <= gradient_tolerance:
            result = dict(success=True, message='Initial point converged', iterations=0)
            energy, gradient = old_energy, old_gradient
        elif method in ('lbfgs','diis'):
            history = self._curvature_history() if method == 'lbfgs' else None
            reused = len(history) if history is not None else 0
            res = minimize_metric(objective,initial,excitation_curvature(self.integrals,self.excitations),
                                  gradient_tolerance=gradient_tolerance,maxiter=maxiter,history=history,
                                  algorithm=method,scale_metric=True,memory=10 if method=='lbfgs' else 8,warmup=2)
            accepted = bool(res.success and np.all(np.isfinite(res.jac)) and np.isfinite(res.fun) and res.fun <= old_energy+energy_tolerance)
            if accepted:
                self.theta = res.x.copy()
                energy, gradient = float(res.fun), res.jac
                pairs = res.history[-10:]
                self._optimizer_state = dict(version=1,context=self._optimizer_context(),ids=[e.id for e in self.excitations],theta=self.theta.tolist(),
                                             s=[s.tolist() for s,y in pairs],y=[y.tolist() for s,y in pairs]) if pairs else None
            else:
                energy, gradient = old_energy, old_gradient
            result = dict(success=accepted,message=str(res.message),iterations=int(res.nit))
        else:
            options = dict(maxiter=maxiter)
            if method.lower() == 'slsqp':
                options['ftol'] = energy_tolerance
            else:
                options.update(gtol=gradient_tolerance,norm=2)
            res = minimize(objective,initial,method=method,jac=True,options=options)
            energy, gradient = objective(res.x) if np.all(np.isfinite(res.x)) else (np.inf,old_gradient)
            accepted = bool(np.isfinite(energy) and np.all(np.isfinite(gradient)) and energy <= old_energy+energy_tolerance and (res.success or np.linalg.norm(gradient) <= gradient_tolerance))
            if accepted:
                self.theta = np.asarray(res.x, dtype=float)
                self._optimizer_state = None
            else:
                energy, gradient = old_energy, old_gradient
            result = dict(success=accepted, message=str(res.message), iterations=int(res.nit))
        result.update(evaluations=evaluations, optimizer=method, reused_curvature_pairs=reused, energy=float(energy), gradient_norm=float(np.linalg.norm(gradient)), **self.resources(), **self.diagnostics())
        self.needs_optimization = not result['success']
        return result

    def _optimize_pqe(self, *, residual_tolerance, maxiter):
        evaluations = 0
        initial = self.theta.copy()
        old_energy, old_residual = self.projective_residuals(initial)
        evaluations += 1
        last = [initial.copy(), (old_energy, old_residual)]
        def objective(parameters):
            nonlocal evaluations
            if not np.array_equal(parameters, last[0]):
                last[:] = [np.array(parameters, copy=True), self.projective_residuals(parameters)]
                evaluations += 1
            return last[1]
        res = solve_pqe(objective, initial, excitation_curvature(self.integrals, self.excitations)/2,
                        residual_tolerance=residual_tolerance, maxiter=maxiter)
        # PQE solves equations; its energy need not decrease on every update.
        if res.success:
            self.theta = res.x.copy()
            self._optimizer_state = None
            energy, residual = float(res.fun), res.residual
        else:
            energy, residual = old_energy, old_residual
        self.needs_optimization = not res.success
        return dict(success=bool(res.success), message=str(res.message), iterations=int(res.nit),
                    evaluations=evaluations, optimizer='pqe', energy=float(energy),
                    residual_norm=float(np.linalg.norm(residual)),
                    **self.resources(), **self.diagnostics())

    def run(self, thresholds=None, *, checkpoint=None, max_cycles=50, max_operators=None, gradient_tolerance=None, residual_tolerance=None, energy_tolerance=None, maxiter=None, optimizer=None, callback=None):
        if thresholds is None:
            thresholds = self.schedule if self.schedule is not None else (1e-2,1e-3,1e-4)
        thresholds = tuple(float(x) for x in thresholds)
        if not thresholds or any(not np.isfinite(x) or x < 0 for x in thresholds) or any(a <= b for a,b in zip(thresholds,thresholds[1:])):
            raise ValueError('Thresholds must be finite, nonnegative, and strictly decreasing')
        if max_cycles < 1 or (max_operators is not None and max_operators < 0):
            raise ValueError('Invalid resource limit')
        if self.schedule is not None and tuple(self.schedule) != thresholds:
            raise ValueError('A restart must use the saved threshold schedule')
        self.schedule = thresholds
        for name, value in dict(gradient_tolerance=gradient_tolerance, residual_tolerance=residual_tolerance, energy_tolerance=energy_tolerance, maxiter=maxiter, method=optimizer).items():
            if value is not None:
                if name in ('method','gradient_tolerance','residual_tolerance') and value != self.optimization_settings.get(name):
                    self.needs_optimization = True
                self.optimization_settings[name] = value
        def record(event):
            self.history.append(event)
            if callback is not None:
                callback(event)
            if checkpoint is not None:
                self.save_checkpoint(checkpoint)
        if not self.initialized:
            if self.seed == 'sd':
                pool = singlet_excitations(self.integrals.electrons, self.qubits, range(1,min(2,self.selector.max_rank)+1), self.selector.orbital_irreps)
                blocks = sorted({e.block for e in pool})
                proposals = [(b,0.,'full_sd_seed') for b in blocks]
            elif self.selection == 'uscc':
                proposals = [(b,s,why) for b,(s,why) in sorted(self.selector.base_scores().items()) if s >= thresholds[0]]
            else:
                proposals = []  # Residual/variational criteria first screen the HF state.
            added, limited = self._append_blocks(proposals, max_operators)
            self.initialized = not limited
            self.needs_optimization = True
            if limited:
                self.stop_reason = 'operator_limit'
                record(dict(event='seed_limited', added=added))
                return self.result()
        cycles = 0
        while True:
            if self.needs_optimization:
                optimization = self.optimize(**self.optimization_settings)
                self.stop_reason = 'running' if optimization['success'] else 'optimizer_failure'
                record(dict(event='optimization', threshold_index=self.threshold_index, **optimization))
                if not optimization['success']:
                    return self.result()
            if self.threshold_index >= len(thresholds):
                self.stop_reason = 'schedule_exhausted'
                record(dict(event='finished', reason=self.stop_reason))
                return self.result()
            if cycles >= max_cycles:
                self.stop_reason = 'cycle_limit'
                record(dict(event='finished', reason=self.stop_reason))
                return self.result()
            epsilon = thresholds[self.threshold_index]
            proposals, selection_diagnostics = self._select(epsilon)
            if not proposals:
                self.threshold_index += 1
                record(dict(event='threshold_closed', epsilon=epsilon, **selection_diagnostics))
                continue
            added, limited = self._append_blocks(proposals, max_operators)
            cycles += 1
            if added:
                self.stop_reason = 'running'
                record(dict(event='growth', epsilon=epsilon, added=added, **selection_diagnostics, **self.resources()))
            if limited:
                self.stop_reason = 'operator_limit'
                record(dict(event='finished', reason=self.stop_reason))
                return self.result()

    def result(self):
        projective = {}
        if self.selection == 'spqe' or self.optimization_settings.get('method') == 'pqe':
            _, vector, indices = self.projective_residual_state()
            residuals = self._reference_projector(indices).project(vector)
            projective = dict(full_residual_norm=float(np.linalg.norm(vector)), residual_norm=float(np.linalg.norm(residuals)))
        return dict(energy=float(self.expectation()), stop_reason=self.stop_reason, selection=dict(method=self.selection, seed=self.seed, threshold_index=self.threshold_index, thresholds=self.schedule, requested_max_rank=self.selector.requested_max_rank, effective_max_rank=self.selector.max_rank), **projective, **self.resources(), **self.diagnostics(), history=list(self.history))

    def _fingerprint(self):
        digest = sha256()
        for array in (self.integrals.one_body, self.integrals.two_body):
            digest.update(np.asarray(array, dtype='<f8').tobytes())
        digest.update(json.dumps(dict(constant=self.integrals.constant, electrons=self.integrals.electrons, metadata=self.metadata), sort_keys=True).encode())
        return digest.hexdigest()

    def save_checkpoint(self, filename):
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        optimizer_state = self._optimizer_state
        if optimizer_state is not None and (optimizer_state['context']['observable'] is not None or self._curvature_history() is None):
            optimizer_state = None
        data = dict(version=FORMAT_VERSION, convention=CONVENTION, fingerprint=self._fingerprint(), integrals=dict(one_body=self.integrals.one_body.tolist(), two_body=self.integrals.two_body.tolist(), constant=self.integrals.constant, electrons=self.integrals.electrons), metadata=self.metadata,
                    config=dict(max_rank=self.selector.requested_max_rank, seed=self.seed, selection=self.selection, trotter_steps=self.trotter_steps, device=self.device_name, orbital_irreps=self.selector.orbital_irreps, backend=self.backend, sector_threads=self.sector_threads, sector_kernels=self.sector_kernels, rotation_threads=self.rotation_threads), excitations=[e.to_dict() for e in self.excitations], ids=[e.id for e in self.excitations], parameters=self.theta.tolist(), history=self.history, provenance=self.provenance, schedule=self.schedule, threshold_index=self.threshold_index, initialized=self.initialized, needs_optimization=self.needs_optimization, stop_reason=self.stop_reason, optimization_settings=self.optimization_settings, optimizer_state=optimizer_state)
        temporary = filename.with_name(filename.name+'.tmp')
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
        temporary.replace(filename)

    @classmethod
    def from_checkpoint(cls, filename, *, device=None, backend=None, sector_threads=None, sector_kernels=None, rotation_threads=None):
        data = json.loads(Path(filename).read_text())
        if data.get('version') != FORMAT_VERSION or data.get('convention') != CONVENTION:
            raise ValueError('Unsupported checkpoint schema or excitation convention')
        config = dict(data['config'])
        config.setdefault('backend','pennylane')  # Legacy checkpoints retain their backend.
        if backend is not None:
            config['backend'] = backend
        if device is not None:
            config['device'] = device
        if sector_threads is not None:
            config['sector_threads'] = sector_threads
        if sector_kernels is not None:
            config['sector_kernels'] = sector_kernels
        if rotation_threads is not None:
            config['rotation_threads'] = rotation_threads
        model = cls(ActiveSpaceIntegrals(**data['integrals']), metadata=data['metadata'], **config)
        if model._fingerprint() != data['fingerprint']:
            raise ValueError('Checkpoint integral/orbital fingerprint mismatch')
        excitations = [Excitation.from_dict(e) for e in data['excitations']]
        if [e.id for e in excitations] != data['ids']:
            raise ValueError('Checkpoint excitation identity mismatch')
        # Reconstruct the canonical scalar groups, rejecting edited definitions.
        for excitation in excitations:
            if excitation not in singlet_excitations_for_block(*excitation.block):
                raise ValueError('Checkpoint contains a noncanonical scalar excitation')
        model.set_excitations(excitations, data['parameters'])
        for name in ('history','provenance','schedule','threshold_index','initialized','needs_optimization','stop_reason','optimization_settings'):
            setattr(model, name, data[name])
        model._optimizer_state = data.get('optimizer_state')
        model._curvature_history(strict=True)
        if not isinstance(model.threshold_index, int) or model.threshold_index < 0 or (model.schedule is not None and model.threshold_index > len(model.schedule)):
            raise ValueError('Invalid checkpoint schedule position')
        return model
