#!/usr/bin/env python

import numpy as np
import gurobipy as gb

ntwk = gb.Model('3-node network')

#################################
# Adding variables to the model #
#################################
# -- Giving infinity a better name --
inf = gb.GRB.INFINITY

# -- Adding the flows --
F12 = ntwk.addVar(lb = -inf, ub = inf, name = 'flow 1->2')
F13 = ntwk.addVar(lb = -inf, ub = inf, name = 'flow 1->3')
F23 = ntwk.addVar(lb = -inf, ub = inf, name = 'flow 2->3')

# -- Adding the mismatches --
mm1 = ntwk.addVar(name = 'mismatch in node 1')
mm2 = ntwk.addVar(name = 'mismatch in node 2')
mm3 = ntwk.addVar(name = 'mismatch in node 3')

mm1 = 6
mm2 = -9
mm3 = -6

# -- Adding the balancing --
b1 = ntwk.addVar(lb = -inf, ub = inf, name = 'balance in node 1')
b2 = ntwk.addVar(lb = -inf, ub = inf, name = 'balance in node 2')
b3 = ntwk.addVar(lb = -inf, ub = inf, name = 'balance in node 3')

# -- Adding the injection patterns --
ip1 = ntwk.addVar(lb = -inf, ub = inf, name = 'node 1 injection pattern')
ip2 = ntwk.addVar(lb = -inf, ub = inf, name = 'node 2 injection pattern')
ip3 = ntwk.addVar(lb = -inf, ub = inf, name = 'node 3 injection pattern')



# -- Remember to update the model after adding variables --
ntwk.update()


################################
# Adding the equations by hand #
################################

# -- Adding step 1 objective function --
ntwk.setObjective(b1*b1 + b2*b2 + b3*b3, sense = gb.GRB.MINIMIZE)

# -- Adding step 1 constraints --
ntwk.addConstr(b1 + ip1 == mm1, name = 'balance in node 1 + node 1 injection pattern = mismatch in node 1')
ntwk.addConstr(b2 + ip2 == mm2, name = 'balance in node 2 + node 2 injection pattern = mismatch in node 2')
ntwk.addConstr(b3 + ip3 == mm3, name = 'balance in node 3 + node 3 injection pattern = mismatch in node 3')

ntwk.addConstr(ip1 + ip2 + ip3 == 0, name = 'node 1 injection pattern + node 2 injection pattern + node 3 injection pattern = 0')

ntwk.optimize()

# -- Adding step 2 objective function --
ntwk.setObjective(F12*F12 + F23*F23 + F13*F13, sense = gb.GRB.MINIMIZE)

# -- Assigning values --
ip1 = ntwk.getVarByName('node 1 injection pattern').X
ip2 = ntwk.getVarByName('node 2 injection pattern').X
ip3 = ntwk.getVarByName('node 3 injection pattern').X

ntwk.update()

# -- Adding step 2 constraints --
ntwk.addConstr(F12 + F13 == ip1, name = 'flow 1->2 + flow 1->3 = %2.2f' % ip1)
ntwk.addConstr(-F12 + F23 == ip2, name = ' - flow 1->2 + flow 2->3 = %2.2f' % ip2)
ntwk.addConstr(-F13 - F23 == ip3, name = ' - flow 1->3 - flow 2->3 = %2.2f' % ip3)

ntwk.optimize()

print 'Variables of the model:'
for var in ntwk.getVars():
    print var.VarName
print
print 'Constraints of the model:'
for constr in ntwk.getConstrs():
    print constr.ConstrName



