#!/usr/bin/env python3

import uccsd
import solvers
import numpy as np

# ----------------------------------------------------------------------

ucc = uccsd.uccsd()
ucc.ground_state()

dipx, dipy, dipz = ucc.property_gradient('int1e_r')
resp_x = solvers.cg(ucc.hvp, dipx, verbose=True)
print('\ndipx: \n', dipx)
print('\nresp_x: \n', resp_x)
print('\npolarisability: \n', np.dot(resp_x, dipx))
