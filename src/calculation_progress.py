"""Optional calculation events and their human-readable CLI presentation.

Library calls are silent unless given a progress callback. These transient
events do not enter numerical histories or checkpoints.
"""


def emit(progress, event, **details):
    if progress is not None:
        progress(dict(event=event, **details))


class ConsoleProgress:
    def __init__(self, scope, *, quiet=False):
        self.scope = scope
        self.quiet = quiet

    def message(self, message):
        if not self.quiet:
            print(f'[{self.scope}] {message}', flush=True)

    def __call__(self, event):
        name = event['event']
        if name == 'scf_start':
            message = f"Solving the RHF reference: basis={event['basis']}, charge={event['charge']}, {event['atoms']} atoms."
        elif name == 'scf_iteration':
            message = (f"RHF iteration {event['iteration']}: E={event['energy']:.12f} hartree, "
                       f"|orbital gradient|={event['gradient_norm']:.3e}.")
        elif name == 'scf_complete':
            message = f"RHF converged: E={event['energy']:.12f} hartree."
        elif name == 'integrals_start':
            message = f"Building active-space integrals: {event['electrons']} electrons in {event['orbitals']} orbitals; {event['core']} frozen core orbitals."
        elif name == 'seed_start':
            message = f"Preparing the {event['seed']} ground-state seed at threshold {event['epsilon']:g}."
        elif name == 'seed_complete':
            message = f"Seed added {event['added']} operators; {event['dimension']} total, ranks={event['ranks']}."
        elif name == 'optimization_start':
            criterion = 'projected residual' if event['optimizer'] == 'pqe' else 'gradient'
            message = (f"Starting {event['optimizer']} optimization of {event['dimension']} amplitudes; "
                       f"L2 {criterion} target={event['tolerance']:.3e}, max iterations={event['maxiter']}.")
        elif name == 'optimization_iteration':
            criterion = 'projected residual' if event['optimizer'] == 'pqe' else 'gradient'
            message = (f"{event['optimizer']} iteration {event['iteration']}: E={event['energy']:.12f} hartree, "
                       f"|{criterion}|={event['norm']:.3e} (target {event['tolerance']:.3e}).")
        elif name == 'optimization':
            norm = event.get('residual_norm', event.get('gradient_norm'))
            status = 'converged' if event['success'] else 'failed; keeping the previous amplitudes'
            message = (f"Optimization {status}: {event['iterations']} iterations, E={event['energy']:.12f}, "
                       f"convergence norm={norm:.3e}; {event['message']}.")
        elif name == 'selection_start':
            message = f"Screening ground-state operators at threshold {event['epsilon']:g} (growth cycle {event['cycle']}, {event['dimension']} current operators)."
        elif name == 'growth_start':
            message = f"About to add {event['operators']} operators at threshold {event['epsilon']:g}."
        elif name == 'growth':
            message = f"Added {len(event['added'])} operators; {sum(event['ranks'].values())} total, ranks={event['ranks']}. Amplitudes will be reoptimized."
        elif name == 'threshold_closed':
            message = f"Threshold {event['epsilon']:g} closed: no more operators passed screening. Advancing the threshold schedule."
        elif name == 'seed_limited':
            message = f"Seed stopped at the operator limit after adding {len(event['added'])} operators."
        elif name == 'finished':
            message = f"Ground-state selection stopped: {event['reason']}."
        elif name == 'pool_start':
            message = f"Building the independent singlet LR candidate space through excitation rank {event['max_rank']}."
        elif name == 'reference_start':
            message = f"Candidate space: {event['dimension']} CSFs; preparing the correlated reference and measuring its residual."
        elif name == 'reference_complete':
            message = f"Reference E={event['energy']:.12f} hartree, ground-state residual={event['residual_norm']:.3e}."
        elif name == 'properties_start':
            message = 'Computing correlated property couplings for the response seed.'
        elif name == 'lr_seed':
            message = f"LR seed contains {event['dimension']}/{event['candidate_dimension']} CSFs ({event['description']})."
        elif name == 'lr_cycle':
            message = f"Selection cycle {event['cycle']+1}: solving in {event['dimension']} CSFs before external residual screening."
        elif name == 'screen_start':
            operator = {'coupling': 'U†HU', 'bare-coupling': 'H', 'linear-coupling': 'H+[H,A]'}[event['method']]
            message = f"Screening external CSF residuals using {operator} ({event['method']})."
        elif name == 'lr_growth':
            if event['added'] and event['will_grow']:
                message = f"About to add {event['added']} CSFs: {event['dimension']} -> {event['dimension']+event['added']}."
            elif event['added']:
                message = f"{event['added']} additional CSFs passed screening; growth limit reached, keeping the last solved space."
            else:
                message = 'No additional CSFs passed screening; selection converged.'
        elif name == 'linear_start':
            message = (f"Solving (K-zI)x=b: omega={event['omega']:g}, gamma={event['gamma']:g} hartree, "
                       f"{event['dimension']} CSFs; relative residual target={event['tolerance']:.3e}, "
                       f"absolute target={event['absolute_tolerance']:.3e}, max iterations={event['maxiter']}.")
        elif name == 'linear_iteration' and event.get('residual_replacement'):
            message = (f"Refreshing the Davidson residual after iteration {event['iteration']}: "
                       f"fresh |residual|={event['residual_norm']:.3e} exceeds target {event['tolerance']:.3e}; "
                       "restarting from the current solution and its audited image.")
        elif name == 'linear_iteration':
            message = (f"Response Davidson iteration {event['iteration']}: |residual|={event['residual_norm']:.3e} "
                       f"(target {event['tolerance']:.3e}), subspace={event['dimension']}.")
        elif name == 'linear_complete':
            message = f"Linear solve converged: checked |residual|={event['residual_norm']:.3e}."
        elif name == 'linear_cached':
            message = f"Reusing a converged solution for omega={event['omega']:g}, gamma={event['gamma']:g}; |residual|={event['residual_norm']:.3e}."
        elif name == 'linear_zero':
            message = 'Property source is zero in this space; the response solution is zero.'
        elif name == 'matrix_progress':
            message = f"LR matrix construction: {event['completed']}/{event['dimension']} columns."
        elif name == 'eigen_start':
            message = (f"Solving for {event['roots']} lowest {event['kind']} roots in dimension {event['dimension']} "
                       f"using {event['solver']}; residual target={event['tolerance']:.3e} hartree, "
                       f"iteration limit={event['maxiter']}.")
        elif name == 'eigen_iteration':
            message = (f"{event['kind']} Davidson iteration {event['iteration']}: "
                       f"max |residual|={max(event['residual_norms']):.3e}, "
                       f"subspace={event['dimension']}/{event['subspace_limit']}, "
                       f"operator products={event['products']}, restarts={event['restarts']}.")
        elif name == 'residual_screen':
            message = (f"External screening completed for {event['images']} vector images; "
                       f"largest CSF residual norm={event['maximum']:.3e}.")
        elif name == 'eigen_work':
            message = f"{event['kind']} eigensolver: {event['products']} operator applications completed."
        elif name == 'eigen_root':
            message = f"{event['kind']} root {event['root']} converged: {event['energy']:.12f} hartree, checked |residual|={event['residual_norm']:.3e}."
        else:
            return
        self.message(message)
