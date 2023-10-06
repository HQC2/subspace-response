#!/usr/bin/env python3

import uccsd
import solvers
import numpy as np
from pyscf.data.gyro import get_nuc_g_factor
from pyscf.data import nist

#-----------------------------------------------------------------------

ucc = uccsd.uccsd()
ucc.ground_state()

ao_int = ucc.m.intor("int1e_ipovlp")
print('\nAO integral:\n', ao_int)
dipx, dipy, dipz = ucc.property_gradient(ao_int)
resp_x = solvers.cg(ucc.hvp, dipx, verbose=True)
print('\ndipx: \n', dipx)
print('\nresp_x: \n', resp_x)
print('\npolarisability: \n', np.dot(resp_x, dipx))
