import numpy as np
from time import time
import gurobipy as gb
from solvers import AtoKh
from solvers import print_status
from copy import deepcopy


def AtoKh_old(N,h0=None,pathadmat=None):
    if pathadmat==None:
        pathadmat=N.pathadmat
    Ad=np.genfromtxt(pathadmat,dtype='d')
    L=0
    listFlows=[]
    for j in range(len(Ad)):
        for i in range(len(Ad)):
            if i>j:
                if Ad[i,j] > 0:
                    L+=1
    K=np.zeros((len(Ad),L))
    h=np.zeros(L*2)
    h=np.append(h,np.zeros(3*len(Ad)))
    L=0
    j=0
    i=0
    for j in range(len(Ad)):
        for i in range(len(Ad)):
            if i>j:
                if Ad[i,j] > 0:
                    K[j,L]=1
                    K[i,L]=-1
                    h[2*L]=Ad[i,j]
                    h[2*L+1]=Ad[j,i]
                    listFlows.append([ str(N[j].label) , str(N[i].label) , L ])
                    L+=1
    if h0 == None:
        h0=h

    return K,h0,listFlows


def build_DC_network(N, copper=False, h0=None, b=1):
    K = AtoKh_old(N)[0]
    H = AtoKh(N)[-2]
    Nlinks = len(AtoKh(N)[-1])
    Nnodes = len(N)

    network = gb.Model('network')

####### Adding variables to the model with the appropriate bounds ####

    netexport_vars = []
    #storage_vars = []
    balancing_vars = []
    curtailment_vars = []

    for i in xrange(Nnodes):
        netexport_vars.append(network.addVar(lb=-gb.GRB.INFINITY,\
                               ub=gb.GRB.INFINITY,\
                               name='phi'+str(i)))
        #storage_vars.append(network.addVar(lb=0.0,\
        #                       ub=gb.GRB.INFINITY,\
        #                       name='s'+str(i)))
        balancing_vars.append(network.addVar(lb=0.0,\
                               ub=gb.GRB.INFINITY,\
                               name='b'+str(i)))
        curtailment_vars.append(network.addVar(lb=0.0,\
                               ub=gb.GRB.INFINITY,\
                               name='c'+str(i)))
    network.update()

####### Adding constraint on netexports making sure total netexports
####### is zero ####################################################

    network.addConstr(gb.quicksum(netexport_vars)==0, \
            'FlowConservation')
    network.update()

###### Adding nodal balancing constraints for each node: #############
###### \phi_n - B_n + C_n = \Delta_n #################################
###### \Delta is initially set to 1e6 but is manually adjusted #######
###### in each time step when the network is solved ##################

    for i in xrange(Nnodes):
        network.addConstr(lhs=netexport_vars[i]\
                               - balancing_vars[i]\
                               + curtailment_vars[i], sense='=', rhs=1e6,
                               name='NodalEnergyBalance'+str(i))

    network.update()

##### Adding constraints on the netexport pattern derived from the ##########
##### constraints on the flows f^-_l <= [K^T L^+ \Phi]_l <= f^+_ l ##########
    if (h0 != None):
        H = h0
    h_neg = -b*H[1:Nlinks*2:2] # lower bound on link flow
    h_pos = b*H[0:Nlinks*2:2] # upper bound on link flow

    L = np.dot(K, K.transpose())
    Lplus = np.linalg.pinv(L)
    KTLplus = np.dot(K.transpose(), Lplus)

    if not copper:
        flows = []
        for l in xrange(Nlinks):
            coeffs = list(KTLplus[l,:])
            flows.append(gb.LinExpr(coeffs, netexport_vars))

        for l in xrange(Nlinks):
            network.addConstr(lhs=flows[l], sense='<=', rhs=h_pos[l],\
                               name='Linkflow_lb'+str(l))
            network.addConstr(lhs=flows[l], sense='>=', rhs=h_neg[l],\
                               name='Linkflow_ub'+str(l))
        network.update()

#### Building the Gurobi expression for the sum of squared flows ###########
    sum_of_squared_flows = gb.QuadExpr()
    for l in xrange(Nlinks):
        for n in xrange(Nnodes):
            for m in xrange(Nnodes):
                sum_of_squared_flows.addTerms(KTLplus[(l,n)]*KTLplus[(l,m)],\
                                              netexport_vars[n],\
                                              netexport_vars[m])

    network.setParam("OutputFlag", 0)
    network.setParam("FeasibilityTol", 1e-6)
    network.update()

    return network, sum_of_squared_flows


def DC_solve(N, mode='linear', h0=None, b=1.0, lapse=None, msg="power flows"):
    """ This function analogous to solve(...) in aurespf.solvers
        It can be called with the same syntax, but does not include storage,
        and the only accepted modes are 'linear', 'square' and
        'square twostep'. 'linear' solves the flows in the localized flow
        paradigm, 'square' solves it in the synchronized paradigm.
        In the unconstrained case this flow implementation is equivalent
        to the one in aurespf.solver, but in constrained networks it yields
        physically correct phase angle flows, unlike the implementation in
        aurespf.solver.

        Example
        -------
        solve(mynodes, mode='linear', h0=myh0, b=0.45)

        """

    print msg
#### Loading in common parameters ############################################
    copper = ('copper' in mode)
    verbose = ('verbose' in mode)

    if lapse==None:
        lapse = np.arange(0, N[0].nhours,1)
    if type(lapse) == int:
        lapse = np.arange(0, lapse, 1)

    Nlinks = len(AtoKh(N)[-1])
    Nnodes = len(N)
    length_of_timeseries = N[0].nhours
    mean_loads = [n.mean for n in N]


    K = AtoKh_old(N)[0] # incidence matrix
    L = np.dot(K, K.transpose()) # Laplacian
    Lplus = np.linalg.pinv(L) # Moore-Penrose pseudo-invers of Laplacian
    KTLplus = np.dot(K.transpose(), Lplus) # Matrix product for finding flows

#### Setting up the model ####################################################
    network, sum_of_squared_flows = build_DC_network(N,\
                                                    copper=copper, h0=h0, b=b)

#### Preparing for collection of the data from the solution of the network ###
    balancing = np.zeros((Nnodes, length_of_timeseries))
    curtailment = np.zeros((Nnodes, length_of_timeseries))
    flow = np.zeros((Nlinks, length_of_timeseries))

#### Solving the network in the specified timesteps ##########################

    calculation_start = time()
    relaxations = 0
    total_tries = 0
    for t in lapse:
        solution, r, tries = _solve_DC_flows_(network, N, t, mode, \
                                            sum_of_squared_flows, mean_loads)
        relaxations += r
        total_tries += tries
        ## change this if implementing storage!!
        balancing[:,t] = solution[1::3]
        curtailment[:,t] = solution[2::3]
        net_export = np.array(solution[0::3])
        flow[:,t] = np.dot(KTLplus, net_export)

        if t>0 and verbose:
            print_status(start=calculation_start, t=t, \
                                l=len(lapse),relaxed=relaxations)
    if verbose:
        print_status(start=calculation_start, t=t, \
                         l=len(lapse),relaxed=relaxations)


#### Saving the results in a Nodes object ####################################
    for n in N:#solved_nodes:
        n.balancing = balancing[n.id, :]
        n.curtailment = curtailment[n.id, :]
        n.solved = True

    T = time() - calculation_start
    print("Calculation of " + msg + " took %2.0f:%02.0f." \
            + "\nNumber of relaxations: %i. Number of extra tries: %i") % \
                    (T/60.0-np.mod(T/60.0,1), np.mod(T,60), relaxations, \
                            total_tries-len(lapse))

    return N, flow


def _solve_DC_flows_(network, N, t, mode, sum_of_squared_flows, mean_loads):

    Nnodes = len(N)
    relaxed = 0
    tries = 0

#### Set the rhs to be the actual mismatch in the NodalEnergyBalance #####
#### for each node #######################################################

    mismatch = [n.mismatch[t] for n in N]
    for i in xrange(len(N)):
        constr_name = 'NodalEnergyBalance' + str(i)
        network.getConstrByName(constr_name).setAttr("rhs", float(mismatch[i]))
    network.update()

#### Set up step 1 objective ##############################################
    balancing_vars = [network.getVarByName('b'+str(i))\
                                              for i in xrange(len(N))]
    curtailment_vars = [network.getVarByName('c'+str(i))\
                                              for i in xrange(len(N))]

    if 'linear' in mode:
        network.setObjective(gb.quicksum(balancing_vars), sense=1)
    elif 'square' in mode:
        sqr_step1_objective = gb.QuadExpr()
        sqr_coeffs = np.ones(Nnodes)/mean_loads
        sqr_step1_objective.addTerms(sqr_coeffs, balancing_vars, balancing_vars)
        sqr_step1_objective.addTerms(sqr_coeffs, \
                                            curtailment_vars, curtailment_vars)
        network.setObjective(sqr_step1_objective, sense=1)
    else:
        print "Error: Mode not understood. Try 'linear' or 'square'"
        return

    network.update()

#### Solve step 1 and save results or objective ###############################
    network.optimize()

    # the synchronized flow (square) only need the first step to uniquely
    # determine the injection pattern and thus the flows
    if 'square' in mode:
        if 'twostep' not in mode:
            try:
                tries += 1
                result = [var.x for var in network.getVars()]
            except gb.GurobiError as error:
                tries += 1
                print "Model status: ", network.status
                print "First step error, timestep: %i" % t
                print type(error)
                print error.errno
                network_relaxed = network.copy()
                network_relaxed.feasRelaxS(0,0,0,1)
                network_relaxed.update()
                relaxed += 1
                network_relaxed.optimize()
                result = [var.x for var in \
                            network_relaxed.getVars()[0:(3*Nnodes)]]
        elif 'twostep' in mode:
            step1_bal = [b.x for b in balancing_vars]
            step1_curt = [c.x for c in curtailment_vars]
            # make sure backup and curtailment matches first step
            for n in range(Nnodes):
                balancing_vars[n].ub = step1_bal[n]*1.00000001
                balancing_vars[n].lb = step1_bal[n]*0.99999999
                curtailment_vars[n].ub = step1_curt[n]*1.00000001
                curtailment_vars[n].lb = step1_curt[n]*0.99999999

            # set step 2 objective, minimize squared flows
            network.setObjective(expr=sum_of_squared_flows, sense=1)
            network.update()
            try:
                tries += 1
                network.optimize()
                result = [var.x for var in network.getVars()]
            except gb.GurobiError:
                try:
                    tries += 1
                    print "Model status: ", network.status
                    network.getConstrs()[-1].setAttr("rhs", bal_opt*1.00001)
                    network.optimize()
                    result = [var.x for var in network.getVars()]
                    relaxed += 1
                except gb.GurobiError as error:
                    tries += 1
                    print "Model status: ", network.status
                    print "Second step error, timestep: %i" % t
                    print type(error)
                    print error.errno
                    network_relaxed = network.copy()
                    network_relaxed.feasRelaxS(0,0,0,1)
                    network_relaxed.update()
                    relaxed += 1
                    network_relaxed.optimize()
                    result = [var.x for var in \
                                network_relaxed.getVars()[0:(3*Nnodes)]]

            # clean up
            for n in range(Nnodes):
                balancing_vars[n].ub = gb.GRB.INFINITY
                balancing_vars[n].lb = 0.0
                curtailment_vars[n].ub = gb.GRB.INFINITY
                curtailment_vars[n].lb = 0.0


    # the localized flow (linear) need a second optimization step to uniquely
    # determine the injection pattern
    elif 'linear' in mode:
        bal_opt = network.objVal

#### Set up step 2 constraints ############################################
        network.addConstr(lhs=gb.quicksum(balancing_vars), sense='<=',\
                           rhs=bal_opt, name="sumofbalconstr")
        network.update()

#### Set up step 2 objective ##############################################
        network.setObjective(expr=sum_of_squared_flows, sense=1)
        network.update()

#### Solve step 2 and save results #######################################
        try:
            tries += 1
            network.optimize()
            result = [var.x for var in network.getVars()]
        except gb.GurobiError:
            try:
                tries += 1
                print "Model status: ", network.status
                network.getConstrs()[-1].setAttr("rhs", bal_opt*1.00001)
                network.optimize()
                result = [var.x for var in network.getVars()]
                relaxed += 1
            except gb.GurobiError as error:
                tries += 1
                print "Model status: ", network.status
                print "Second step error, timestep: %i" % t
                print type(error)
                print error.errno
                network_relaxed = network.copy()
                network_relaxed.feasRelaxS(0,0,0,1)
                network_relaxed.update()
                relaxed += 1
                network_relaxed.optimize()
                result = [var.x for var in network_relaxed.getVars()[0:(3*Nnodes)]]

#### Clean up ###########################################################
        network.remove(network.getConstrs()[-1])
        network.update()

    else:
        print "Error: Mode not understood, network not solved.\
                Try 'linear' or 'square'"
        return

    return result, relaxed, tries
