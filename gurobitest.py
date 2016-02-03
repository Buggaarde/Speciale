#!/usr/bin/env python

import numpy as np
import gurobipy as gb

ntwk = gb.Model('3-node network')

#################################
# Adding variables to the model #
#################################

# -- Adding the flows --
F12 = ntwk.addVar(name = 'flow 1->2')
F13 = ntwk.addVar(name = 'flow 1->3')
F23 = ntwk.addVar(name = 'flow 2->3')

# -- Adding the mismatches --
mm1 = ntwk.addVar(name = 'mismatch in node 1')
mm2 = ntwk.addVar(name = 'mismatch in node 2')
mm3 = ntwk.addVar(name = 'mismatch in node 3')

# -- Adding the balance that results from the flow --
mm1_new = ntwk.addVar(name = 'balance after flow in node 1')
mm2_new = ntwk.addVar(name = 'balance after flow in node 2')
mm3_new = ntwk.addVar(name = 'balance after flow in node 3')

# -- Remember to update the model after adding variables --
ntwk.update()


################################
# Adding the equations by hand #
################################

# -- Setting the objective --
# We want to minimize the sum of squared flows, since this corresponds to
# getting flows according to Kirchoff-rules.
ntwk.setObjective(F12*F12 + F13*F13 + F23*F23, sense = gb.GRB.MINIMIZE)

# -- Adding linear flow constraints --
ntwk.addConstr(mm1 - mm1_new == F12 + F13, name = 'node 1')
ntwk.addConstr(mm2 - mm2_new == -F12 + F23, name = 'node 2')
ntwk.addConstr(mm3 - mm3_new == -F13 - F23, name = 'node 3')
mm1 = 6.0
mm2 = -6.0
mm3 = -9.0
# -- Adding linear constraints on the balance --
# These constraints set the initial values for the mismatches
#ntwk.addConstr(mm1 == 6, name = 'value of mismatch in node 1')
#ntwk.addConstr(mm2 == -6, name = 'value of mismatch in node 2')
#ntwk.addConstr(mm3 == -9, name = 'value of mismatch in node 3')

# This particular constraint causes the flow to be localized
#ntwk.addConstr(mm1 + mm2 + mm3 == mm1_new + mm2_new + mm3_new, name = 'mismatch')
ntwk.addConstr(F12 <= -1, name = 'flow larger than one')
#ntwk.addConstr(F23 <= -1, name = 'flow smaller than one')

##################
# Ready to solve #
##################
ntwk.params.DualReductions = 0
ntwk.optimize()
if ntwk.Status == gb.GRB.OPTIMAL:
    vars = ntwk.getVars()
    for v in vars:
        print v
else:
    ntwk.computeIIS()
    ntwk.write('ntwk.ilp')

