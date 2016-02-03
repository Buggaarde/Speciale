#! /usr/bin/env python
from time import time
import sys, os
import random
import gurobipy as gb
import numpy as np
from copy import deepcopy as dcopy
import regions.tools as tls

def AtoKh(N):
    """ For better understanding, go open admat.txt file in the settings folder.
        This function transforms an adjacency table A into the incidence matrix K
        of size N x L with K[n,l] if link 'l' starts at node and -1 if it ends
        there. Also returns 'h', which holds the actual transmission limits.

        In this version, K is returned as row and column indices and values, and
        used to build the problem in gurobi. The final entry returns a list of
        links with names, which is very useful when you get lost in the numbers."""

    Ad=np.genfromtxt(N.pathadmat,dtype='d')
    
    L=0                                 ## A Counter for the number of links
    listFlows=[]
    for j in range(len(Ad)):            ## Scanning the matrix columnwise
        for i in range(len(Ad)):        ## and row wise
            if i>j:                     ## if we are in the lower triangle
                if Ad[i,j] > 0:         ## and there is a link
                    L+=1                ## increase the number of links
    
    K_values=[]                         ## here we define a sparse matrix K
    K_column_indices=[]
    K_row_indices=[]
    T_caps=np.zeros(L*2)
    L=0                                 ## Wow, this is stupid... there must be
    for j in range(len(Ad)):            ## a better way of doing this than repeating
        for i in range(len(Ad)):        ## the whole thing... anyways:
            if i>j:
                if Ad[i,j] > 0:         ## if there is a link
                    K_values.extend([1,-1]) ## add a 1 and a -1 to the values of K
                    K_column_indices.extend([L,L])  ## placed in column L
                    K_row_indices.extend([j,i])     ## and in rows j and i
                    T_caps[2*L]=Ad[i,j]             ## with this capacity for j to i
                    T_caps[2*L+1]=Ad[j,i]           ## and the one on the other side of the triangle
                    
                    listFlows.append([str(N[j].label)+" to " +str(N[i].label), L])
                    L+=1
    return K_row_indices,K_column_indices,K_values,T_caps, listFlows

###############################################################
###############################################################
###############################################################
###############################################################
###############################################################



def print_status(start=0,t=1,l=1000000,relaxed=0,init = False, msg="power flows"):
    """ Internal function that prints a message for every timestep"""
    a=time()-start
    b=l*a*1.0
    if init:
        (a,b)=(0.0,0.0)
        print "Building problem for", msg,"\n\n\n\n"
    print "\033[4A\r\033[0mRelaxations performed:    \033[34m%2.0f" %(relaxed)
    print "\033[0mProgress:    \033[34m%2.2f" %(t*100.0/l)+"%"
    print "\033[0mElapsed time \033[34m%2.0i:%02.0i" % (a/60,np.mod(a,60))
    print "\033[0mETA \033[34m%2.0i:%02.0i " % ((b/t - a)/60,np.mod(b/t-a,60)),
    print "(%2.0i:%02.0i)          " % ((b/t)/60,np.mod(b/t,60))
    sys.stdout.flush()

###############################################################
###############################################################
###############################################################
###############################################################
###############################################################


def build_network(N, copper = 0, h0 = None, b=1):
    """ Internal function that receives a Nodes object with a link to the
        admittance matrix connected to that Nodes object and information
        of the initian transmission constraints (copper, h0, b). It builds
        and returns a gurobi model based on the supplied network."""

    K_row,K_col,K_val,H,LF=AtoKh(N)         ## Admittance matrix and link capacities
    Nlinks=len(K_row)/2                     ## are extracted from N
    Nnodes=len(N)

    if (h0 != None):                        ## If no custom capacities are supplied
        H=h0                                ## by h0, default (current) values are used
    h_neg=b*-H[1:Nlinks*2:2]                ## and set as upper and lower bounds for
    h_pos=b*H[0:Nlinks*2:2]                 ## the flow variables

    network = gb.Model()                    ## An empty network is initialized
                                            ## together with the following variables
    f_names=['f'+str(i+1) for i in range(Nlinks)]   ## flows for L links
    s_names=['s'+str(i+1) for i in range(Nnodes)]   ## storage for N nodes
    b_names=['b'+str(i+1) for i in range(Nnodes)]   ## balancing for N nodes
    c_names=['c'+str(i+1) for i in range(Nnodes)]   ## curtailment for N nodes
    names = f_names + s_names + b_names + c_names

    # Initial upper and lower bounds are stated and collected
    h_lower=h_neg
    h_upper=h_pos
    s_upper=np.ones(Nnodes)*1e9
    s_lower=np.ones(Nnodes)
    b_upper=np.ones(Nnodes)*1e9
    b_lower=np.zeros(Nnodes)                ## note the 0 lower bound for balancing
    c_upper=np.ones(Nnodes)*1e9             ## and curtailment. This should never be
    c_lower=np.zeros(Nnodes)                ## changed
    u_bound=np.concatenate((h_upper,s_upper,b_upper,c_upper))
    l_bound=np.concatenate((h_lower,s_lower,b_lower,c_lower))

    ## variables are added to the model according to 'names'
    if not copper:                          ## for non-copper scenarios, we simply assign
        for i in range(len(names)):         ## bounds to all variables
            network.addVar(lb=l_bound[i],ub=u_bound[i],name=names[i])
    if copper:                              ## but in the case of ''copper'', an exception is
        for i in np.arange(0,Nlinks,1):     ## made for flow variables
            network.addVar(lb = -gb.GRB.INFINITY, ub = gb.GRB.INFINITY, name=names[i])
        for i in np.arange(Nlinks,Nlinks+3*Nnodes,1):
            network.addVar(lb=l_bound[i],ub=u_bound[i],name=names[i])


    network.update()

    a_vars = network.getVars()
### For reference in the rest of the code
###############
#   f_vars = a_vars[:Nlinks]
#   s_vars = a_vars[Nlinks:Nlinks+Nnodes]
#   b_vars = a_vars[Nlinks+Nnodes:-Nnodes]
#   c_vars = a_vars[-Nnodes:]
########################################
    for n in N:  ## Restricts non-load nodes to not balance (e.g. Nort-sea wind farms)
        if np.average(n.load) <= 1e-10:     ## they are detected by their absence of load
            a_vars[Nlinks+Nnodes:-Nnodes][n.id].ub = 0.0    ## but could have an own flag, of course

    network.update()

################### This part states the node equation
################### B - C + S + Imports - Exports = -Delta
################### where Imports-Exports comes from K \cdot f

    for n in range(Nnodes):
        ind=[]
        val=[]
        for i in range(len(K_row)):
            if K_row[i]==n:
                ind.append('f'+str(K_col[i]+1))
                val.append(K_val[i])
        ind.append('s'+str(n+1))
        val.append(-1.0)
        ind.append('b'+str(n+1))
        ind.append('c'+str(n+1))
        val.append(-1.0)
        val.append( 1.0)
        var=[]
        for i in ind:
            var.append(network.getVarByName(i))
        network.addConstr(lhs=gb.LinExpr(val,var),sense='=',rhs=1e6)
        #### 'Delta' is initially set as 1e6, but is changed routinely for every timestep

######################### Setting default values for the solver
###############################################################

    network.setParam("OutputFlag",0)  ## Change this to 1 for debugging individual time steps
    network.setParam("FeasibilityTol",1e-4)
    network.update()
    return network, names, a_vars

###############################################################
###############################################################
###############################################################
###############################################################

def solve(N, mode = "linear", h0=None, b=1.0, lapse=None, msg="power flows"):
    """ This main function builds the network problem, gathers variables and runs
        the time series. It compiles the results and returns them as an Nodes
        object and a Nlinks x Nhours flow matrix.

        mode is a string used to determine the way the solver is to be run. It can
        contain any length of words and commands in any order. Some examples are
        shown below

            linear : run the linear balancing minimisation (no sharing of b)
            square : square balancing minimisation (sharing of b and c)
            hybrid : square balancing minimisation with linear c (sharing of b)
            capped : relaxable balancing constraints (sharing of b)
            storage: needed to ensure s charges (punished c, not needed in square)
            copper : no transmission constraints
            verbose: print solution status at every time step
        example M,F = solve(N,mode = 'copper linear verbose')

        h0 can receive a customized interconnector capacity vector of size
        (2*Nlinks). Else it uses the capacities saved in the admat matrix, part
        of the Nodes object. Default is None.

        b is a linear multiplication factor for whatever interconnector capacities,
        either the default ones or the ones defined by h0. Default is 1.

        lapse defines the timespan to run the simulation, starting at 0. Default is
        None and reverts to Nhours.

        msg is a message printed at the beggining and end of the solution, useful
        when running parallel processes or sweeps over many variables to identify
        specific runs. DOES get printed even without verbose.
    """

###############################################################
################ Loading Default values #######################
# We read copper and verbose form the 'mode' variable, and    #
# the number of links and nodes from AtoKh(N). F, S, B and C  #
# are empty recipients with length equal to the time lapse    #
###############################################################

    verbose = 0
    copper = 0
    if lapse == None:
        lapse = np.arange(0,N[0].nhours,1)
    if type(lapse) == int:
        lapse = np.arange(0,lapse,1)
    if type(lapse) == list:
        lapse = np.arange(lapse[0], lapse[1])

    if 'copper' in mode:
        copper = 1
    if 'verbose' in mode:
        verbose = 1
    if not verbose: print str(msg)
    K_row,K_col,K_val,H,LF=AtoKh(N)
    Nlinks=len(K_row)/2
    Nnodes=len(N)
    l = N[0].nhours
    M = np.zeros((Nnodes,Nlinks))
    for i in range(len(K_row)):
        M[K_row[i]][K_col[i]]=K_val[i] # M is adjacency matrix 'K'

    ### Packing up the sparse adjacency matrix
    ADJ = (M, K_row, K_col, K_val)

    F = np.zeros((Nlinks,l)) #Flows
    S = np.zeros((Nnodes,l)) #Storage
    B = np.zeros((Nnodes,l)) #Balancing
    C = np.zeros((Nnodes,l)) #Curtailment

###############################################################
################# Setting up the model ########################
###############################################################

    network, names, a_vars = build_network(N, copper, h0, b)
    mean_n=[node.mean for node in N]

###############################################################
###### Initial balancing constraints for capped's case ########
# Weights is just the mean load of a country n divided by the #
# mean load of the sum of the nodes. Global_B is the total    #
# balancing capacity required,  which is distributed by       #
# weights to each node as an upper bound. This part can easily#
# be changed to assign different balancing capacities.        #
###############################################################
    if 'RND' in mode:	
        for n in N:
            n.mismatchete=np.zeros(len(lapse))	
            n.temp_mismatch=np.zeros(len(lapse))
            n.imports = np.zeros(len(lapse))	
            n.exports = np.zeros(len(lapse))	
        for t in lapse:
            sinks=[]
            sources=[]
            sur_or_def=np.zeros(len(lapse))	
            sur_or_def[t]=sum(n.mismatch[t] for n in N) # tells us whether the global european network has a total surplus or deficit of RES in the given hour.
            for n in N:	
                if n.mismatch[t]>0:
                    sources.append(n)
                elif n.mismatch[t]<0:
                    sinks.append(n)
            si=len(sinks)
            so=len(sources)	
            n.temp_mismatch[t]=n.mismatch[t]
            intervalssi=np.linspace(0,1,si+1,endpoint=True)
            intervalsso=np.linspace(0,1,so+1,endpoint=True)
            if sur_or_def[t]<=0: #global deficit
                while not all(n.temp_mismatch[t]==0 for n in sources):	
                    rsi=random.random()
                    rso=random.random()
                    for i in range(si):
                        if rsi >=intervalssi[i] and rsi<intervalssi[i+1]:	
                            if (sinks[i].temp_mismatch[t] <> 0):
                                for o in range(so):
                                    if rso >=intervalsso[o] and rso<intervalsso[o+1]:
                                        if (sources[o].temp_mismatch[t] <> 0):	
                                            trade=min(sources[o].temp_mismatch[t],-sinks[i].temp_mismatch[t],1000)	
                                            sources[o].temp_mismatch[t]-=trade
                                            sinks[i].temp_mismatch[t]+=trade	
            elif sur_or_def[t]>0: #global surplus
                while not all(n.temp_mismatch[t] == 0 for n in sinks):	
                    rsi=random.random()
                    rso=random.random()
                    for i in range(si):	
                        if rsi >=intervalssi[i] and rsi<intervalssi[i+1]:	
                            if (sinks[i].temp_mismatch[t] <> 0):	
                                for o in range(so):	
                                    if rso >=intervalsso[o] and rso<intervalsso[o+1]:
                                        if (sources[o].temp_mismatch[t] <> 0):
                                            trade=min(sources[o].temp_mismatch[t],-sinks[i].temp_mismatch[t],1000)	
                                            sources[o].temp_mismatch[t]-=trade
                                            sinks[i].temp_mismatch[t]+=trade
        for n in N:	
            n.mismatchete[t]=n.mismatch[t]-n.temp_mismatch[t] #the mismatch that will actually be dealt with
            n.imports[t]=-min(n.mismatchete[t],0) #the part of the negative mismatches that will be covered by trading
            n.exports[t]=max(n.mismatchete[t],0) #the part of the positive mismatches that will be exported by trading.


    if 'capped' in mode:
        weights = mean_n/sum(mean_n)
        Global_B = np.max( -sum(n.mismatch for n in N) )
        weights *= Global_B*1.001
        for x in range(Nnodes):
            a_vars[Nlinks+Nnodes:-Nnodes][x].ub = float(weights[x])
    network.update()

#   Used in the case of randomised ensembles of Rolando's capped flow. Uses
#   means of balancing capacities in the ensemble as new upper bounds.
    if (('random' in mode) and ('mean' in mode) and ('rolando' in mode)):
        ensembleBc = np.load('./results/bc-ensemble-mean-b-'+str(b)+'.npz')
        for x in range(Nnodes):
            a_vars[Nlinks+Nnodes:-Nnodes][x].ub = float(ensembleBc['Bc'][x])
    network.update()

###############################################################
################# Run the time series  ########################
# The modes 'hybrid' and 'capped' have their own subroutines, #
# all other modes are simply resolved in _solve_flows_. The   #
# status is printed at each time step if verbose. The empty   #
# collectors F, S, B and C are filled as time goes by and only#
# assigned to their resepctive nodes at the end (this is      #
# faster than assigning them to the nodes at each time step)  #
###############################################################


    wholecalcstart = time()

    numberofruns = 1
    if (('capped' in mode) and ('random' not in mode)):
        numberofruns = 2

    for run in range(numberofruns):
        runstart=time()
        relaxed = 0
        if verbose: print_status(init = True)

        for t in lapse:

            if ('hybrid' not in mode) and ('capped' not in mode):
                solution, r = _solve_flows_(network, N, t, ADJ, mode, mean_n)
                relaxed += r

            elif 'hybrid' in mode:
                solution, r = _hybrid_solve_(network, N, t, ADJ, mean_n)
                relaxed += r

            elif 'capped' in mode:
                solution, r = _capped_solve_(network, N, t, ADJ, mean_n, mode)
                relaxed +=r

            ###### save hourly solutions in long-term matrices
            F[:,t]=solution[0:Nlinks]
            S[:,t]=solution[Nlinks:Nlinks+Nnodes]
            B[:,t]=solution[Nlinks+Nnodes:Nlinks+Nnodes*2]
            C[:,t]=solution[-Nnodes:]
            for i in N:
                i.ESS.charge(-S[i.id,t]) # The sign of S is opposite to the sign
                                         # convention used inside the storage class.
                i.balancing[t] = B[i.id,t]
                i.curtailment[t] = C[i.id,t]
                i.storage_discharge[t] = S[i.id,t]
            if t>0 and verbose:
                print_status(start=runstart,t=t,l=l,relaxed=relaxed)
        if verbose: print_status(start=runstart,t=t,l=l,relaxed=relaxed)

    for i in N:
        i._update_()
        i.solved = True
    T=time()-wholecalcstart

    ### save the upper bounds of the balancing capacity
    if 'capped' in mode:
        for n in N:
            n.balancing_upper_bound = a_vars[Nlinks+Nnodes:-Nnodes][n.id].ub

    print ("Calculation of "+msg+" took %2.0f:%02.0f." \
            " %d relaxations where made.") % (T/60.0-np.mod(t/60.0,1),np.mod(T,60), relaxed)

    return dcopy(N),dcopy(F)


###############################################################
###############################################################
###############################################################
###############################################################


def _solve_flows_(network,N,t, ADJ, mode, mean_n):
    """ Non-interactive function, receives a network, a timestep, a bunch of other
        stuff :D and solves both steps of the optimisation process.
    """

    #### Unpacking the sparse matrix
    (M , K_row, K_col, K_val) = ADJ


    relaxed = 0
###### Build variable names ######
##################################
    Nlinks=len(K_row)/2
    Nnodes=len(N)
    a_vars = network.getVars()
    f_vars = a_vars[:Nlinks]
    s_vars = a_vars[Nlinks:Nlinks+Nnodes]
    b_vars = a_vars[Nlinks+Nnodes:-Nnodes]
    c_vars = a_vars[-Nnodes:]

#### Get mismatch & ESS power ####
##################################

    Delta=[i.mismatch[t] for i in N]
    if 'RND' in mode:
        Delta=[n.mismatchete[t] for n in N]
    s_upper=np.array(N.get_P_discharge(t)) ## NOTE: This may be very wrong!
    s_lower=np.array(-N.get_P_charge(t)) ## NOTE: This may be very wrong!
    for n in range(Nnodes):
        if 'smart' in mode:
            if Delta[n] < 0:
                s_lower[n] = min(s_upper[n],-(N.storage_factor)*Delta[n])
                s_upper[n] = min(s_upper[n],-(N.storage_factor)*Delta[n])
        s_vars[n].ub=float(s_upper[n])
        s_vars[n].lb=float(s_lower[n])


#### Set deltas for node equation   ####
########################################
    r=0
    for bc_constraint in network.getConstrs():
        bc_constraint.setAttr("rhs",float(Delta[r]))
        r+=1

    network.update()

#### Set objective factors based on mode ####
#############################################

    if 'linear' in mode or 'capped' in mode or 'RND' in mode:
        b1 = np.ones(Nnodes)
        c1 = np.zeros(Nnodes)
        b2 = np.zeros(Nnodes)
        c2 = np.zeros(Nnodes)
        for n in range(Nnodes):
            c_vars[n].ub = 1e-3
            if Delta[n] > 0:
                c_vars[n].ub = np.float(Delta[n])*(1+1e-3)
        if 'storage' in mode:
            c1 = np.ones(Nnodes)

    if 'square' in mode: 
        b1 = np.zeros(Nnodes)
        c1 = np.zeros(Nnodes)
        b2 = np.ones(Nnodes)/mean_n
        c2 = np.ones(Nnodes)/mean_n

    linear_factors = np.concatenate((b1 , c1))
    square_factors = np.concatenate((b2 , c2))


##### Define Step-1 objective #####
###################################
    step1_objective=gb.QuadExpr()
    step1_objective.addTerms(linear_factors, b_vars+c_vars)
    step1_objective.addTerms(square_factors, b_vars+c_vars, b_vars+c_vars)
    if 'squarestorage' in mode:
        step1_objective.addTerms(0.1*np.ones(Nnodes)/mean_n,s_vars)
    network.setObjective(expr=step1_objective, sense=1)


##### Solve Step-1 objective #########
######################################
    network.update()
    network.optimize()
    v=[]

    for i in network.getVars():
        v.append(i.x)
    BC_opt=network.objVal

##### Add constraints for Step-2 #####
######################################
    if 'square' in mode:
        step2_cs = 0
        b = v[Nlinks+Nnodes:-Nnodes]
        c = v[-Nnodes:]
        s = v[Nlinks:Nlinks+Nnodes]
        for n in range(Nnodes):
            b_vars[n].ub = b[n]*1.00000001
            b_vars[n].lb = b[n]*0.99999999

            c_vars[n].ub = c[n]*1.00000001
            c_vars[n].lb = c[n]*0.99999999
            if 'squarestorage' in mode:
                s_vars[n].ub = s[n]+0.009
                s_vars[n].lb = s[n]-0.009

    else:
        step2_cs = 1
        step2_constraint=gb.LinExpr()
        step2_constraint.addTerms(linear_factors, b_vars+c_vars)
        network.addConstr(lhs=step2_constraint, sense="<", rhs=BC_opt, name="Step 2 constraint")
    network.update()

###### Set Step-2 Objective ######
##################################
    step2_objective=gb.QuadExpr()
    linecost = np.ones(Nlinks)
    if 'impedance' in mode:
        linecost = 0.28*np.array(tls.linfo(N.pathadmat)[:,2], dtype='float')
        assert(len(linecost)==Nlinks), "Make sure there is a suitable lineinfo-file."
    step2_objective.addTerms(linecost, f_vars, f_vars)
    network.setObjective(expr=step2_objective,sense=1)


###### Solve Step 2 ######
##########################

    network.update()
    v=[]
    try:
        v=[]
        network.optimize()
        for i in network.getVars():
            v.append(i.x)
    except gb.GurobiError:
        print "second step error"
        print t
        relaxed += 1
        v=[]
        networkRelaxed=network.copy()
        if 'impedance' in mode:
            networkRelaxed.getConstrByName("Step 2 constraint").setAttr("rhs", BC_opt*1.1)
            networkRelaxed.update()
        else:
            networkRelaxed.feasRelaxS(0,0,0,1)
        networkRelaxed.optimize()
        for i in networkRelaxed.getVars():
            v.append(i.x)


###### Clean-up ######
##########################
    for c in range(step2_cs):
        network.remove(network.getConstrs()[-1])
        network.update()
    if 'square' in mode:
        for n in range(Nnodes):
            b_vars[n].ub = gb.GRB.INFINITY
            b_vars[n].lb = 0.0
            c_vars[n].ub = gb.GRB.INFINITY
            c_vars[n].lb = 0.0
        network.update()
    return v, relaxed


def _hybrid_solve_(network, N, t, ADJ, mean_n):
    mode = "linear"
    relaxed = 0
    solution, r = _solve_flows_(network, N, t, ADJ, mode, mean_n)
    relaxed+= r
    curt_const = solution[-len(N):]  #This is the Y value of solution
    for n in range(len(N)):
        network.getVars()[-len(N):][n].ub = curt_const[n]
    network.update()
    mode = "square"
    solution, r = _solve_flows_(network, N, t, ADJ, mode, mean_n)
    relaxed +=r
    mode = "hybrid"
    for n in range(len(N)):
        network.getVars()[-len(N):][n].ub = gb.GRB.INFINITY
    network.update()
    return solution, relaxed

def _capped_solve_(network, N, t, ADJ, mean_n, mode):

    if ('martin' not in mode) and ('rolando' not in mode):
        raise Exception('Mode not recognized, try capped rolando or capped martin')

    (M , K_row, K_col, K_val) = ADJ
    Nlinks=len(K_row)/2
    Nnodes=len(N)
    b_vars = network.getVars()[Nlinks+Nnodes:-Nnodes]
    relaxed = 0
    converged = 0
    nonconv = []
    preweight=[]
    initialBc = [b_vars[x].ub for x in range(Nnodes)]

    network.update()
    while not converged:
        try:

            solvermode = "capped"
            solution, r =_solve_flows_(network, N, t, ADJ, solvermode, mean_n)
            converged = 1

        except gb.GurobiError:
            if 'rolando' in mode:
                network.computeIIS()
                for x in range(Nnodes):
                    if b_vars[x].getAttr("IISUB"):
                        nonconv.append(x)
                        preweight.append(b_vars[x].ub)
                        b_vars[x].ub += 1.0
                        b_vars[x].ub *= 1.001

            elif 'martin' in mode:
                for x in range(Nnodes):
                    preweight.append(b_vars[x].ub)
                    b_vars[x].ub *= 1.001

            for const in network.getConstrs()[Nnodes:]:
                network.remove(const)

            network.update()
            relaxed += 1
    if 'rolando' in mode:
        for n in range(len(nonconv)):
            b_vars[nonconv[n]].ub=preweight[n]
            a = dcopy(float(solution[Nlinks+Nnodes:-Nnodes][nonconv[n]]))
            if a>preweight[n]:
                b_vars[nonconv[n]].ub=a

    network.update()
    if relaxed>=1e9:
        print "After time = ",t
        raw_input()
        for n in nonconv:
            print N[n].id, N[n].label, N[n].mismatch[t], solution[Nlinks+Nnodes:-Nnodes][n], b_vars[n].ub


    return solution, relaxed


###############################################################
###############################################################
###############################################################
###############################################################

def op_solve(N,mode = 'linear', h0 = None, b=1.0, lapse = 100, msg = "power flows"):
    """ This main function builds the network problem, gathers variables and runs
        the time series. It compiles the results and returns them as an Nodes
        object and a Nlinks x Nhours flow matrix.

        mode is a string used to determine the way the solver is to be run. It can
        contain any length of words and commands in any order. Some examples are
        shown below

            linear : run the linear balancing minimisation (no sharing of b)
            square : square balancing minimisation (sharing of b and c)
            hybrid : square balancing minimisation with linear c (sharing of b)
            capped : relaxable balancing constraints (sharing of b)
            storage: needed to ensure s charges (punished c, not needed in square)
            copper : no transmission constraints
            verbose: print solution status at every time step
        example M,F = solve(N,mode = 'copper linear verbose')

        h0 can receive a customized interconnector capacity vector of size
        (2*Nlinks). Else it uses the capacities saved in the admat matrix, part
        of the Nodes object. Default is None.

        b is a linear multiplication factor for whatever interconnector capacities,
        either the default ones or the ones defined by h0. Default is 1.

        lapse defines the timespan to run the simulation, starting at 0. Default is
        None and reverts to Nhours.

        msg is a message printed at the beggining and end of the solution, useful
        when running parallel processes or sweeps over many variables to identify
        specific runs. DOES get printed even without verbose.
    """

###############################################################
################ Loading Default values #######################
# We read copper and verbose form the 'mode' variable, and    #
# the number of links and nodes from AtoKh(N). F, S, B and C  #
# are empty recipients with length equal to the time lapse    #
###############################################################
    start=time()
    verbose = 0
    copper = 0
    if lapse == None:
        lapse = np.arange(0,N[0].nhours,1)
    if type(lapse) == int:
        lapse = np.arange(0,lapse,1)
    if 'copper' in mode:
        copper = 1
    if 'verbose' in mode:
        verbose = 1
    if not verbose: print str(msg)
    K_row,K_col,K_val,H,LF=AtoKh(N)
    Nlinks=len(K_row)/2
    Nnodes=len(N)
    l = N[0].nhours
    M = np.zeros((Nnodes,Nlinks))
    for i in range(len(K_row)):
        M[K_row[i]][K_col[i]]=K_val[i] # M is adjacency matrix 'K'

    ### Packing up the sparse adjacency matrix
    ADJ = (M, K_row, K_col, K_val)

    T0=time()-start
    T=time()-start
    print "Reading parameters took %2.0f:%02.0f" % (T/60.0-np.mod(T/60.0,1),np.mod(T,60))

    F = np.zeros((Nlinks,l)) #Flows
    S = np.zeros((Nnodes,l)) #Storage
    B = np.zeros((Nnodes,l)) #Balancing
    C = np.zeros((Nnodes,l)) #Curtailment

    if (h0 != None):                        ## If no custom capacities are supplied
        H=h0                                ## by h0, default (current) values are used
    h_neg=b*-H[1:Nlinks*2:2]                ## and set as upper and lower bounds for
    h_pos=b*H[0:Nlinks*2:2]                 ## the flow variables

    network = gb.Model()                    ## An empty network is initialized
                                            ## together with the following variables
    f_names=['f'+str(i)+'_' for i in range(Nlinks)]   ## flows for L links
    s_names=['s'+str(i)+'_' for i in range(Nnodes)]   ## storage for N nodes
    b_names=['b'+str(i)+'_' for i in range(Nnodes)]   ## balancing for N nodes
    c_names=['c'+str(i)+'_' for i in range(Nnodes)]   ## curtailment for N nodes
    names = f_names + s_names + b_names + c_names

    ## loong names###
    NAMES = []
    for t in lapse:
        NAMES.extend([n+str(t) for n in names])
    # Initial upper and lower bounds are stated and collected
    h_lower=h_neg
    h_upper=h_pos
    if copper:
        h_lower = np.ones(Nlinks)*-gb.GRB.INFINITY
        h_upper = np.ones(Nlinks)*gb.GRB.INFINITY
    s_upper=np.array([6*n.mean for n in N])
    s_lower=np.zeros(Nnodes)
    b_upper=np.ones(Nnodes)*1e9
    b_lower=np.zeros(Nnodes)                ## note the 0 lower bound for balancing
    c_upper=np.ones(Nnodes)*1e9             ## and curtailment. This should never be
    c_lower=np.zeros(Nnodes)                ## changed
    u_bound=np.concatenate((h_upper,s_upper,b_upper,c_upper))
    l_bound=np.concatenate((h_lower,s_lower,b_lower,c_lower))

    U_BOUNDS = u_bound
    L_BOUNDS = l_bound
    for t in range(len(lapse)-1):
        U_BOUNDS = np.concatenate((U_BOUNDS,u_bound))
        L_BOUNDS = np.concatenate((L_BOUNDS,l_bound))


    for i in range(len(NAMES)):
        network.addVar(lb=L_BOUNDS[i],ub=U_BOUNDS[i],name=NAMES[i])
    if 'storage' in mode:
        for n in N:
            network.addVar(lb=s_lower[n.id],ub=s_upper[n.id],name='s'+str(n.id)+'_'+str(lapse[-1]+1))

    T0+=T
    T=time()-start-T0
    print "Building variables took %2.0f:%02.0f" % (T/60.0-np.mod(T/60.0,1),np.mod(T,60))

    network.update()

    a_vars = network.getVars()
### For reference in the rest of the code
###############
#   f_vars = a_vars[:Nlinks]
#   s_vars = a_vars[Nlinks:Nlinks+Nnodes]
#   b_vars = a_vars[Nlinks+Nnodes:-Nnodes]
#   c_vars = a_vars[-Nnodes:]
########################################
    for n in N:  ## Restricts non-load nodes to not balance (e.g. Nort-sea wind farms)
        if np.average(n.load) <= 1e-10: ## they are detected by their absence of load
            for t in lapse: ## but could have an own flag, of course
                network.getVarByName('b'+str(n.id)+'_'+str(t)).ub=0.0


    network.update()

################### This part states the node equation
################### B - C + S + Imports - Exports = -Delta
################### where Imports-Exports comes from K \cdot f
    for t in lapse:
        for n in range(Nnodes):
            ind=[]
            val=[]
            for i in range(len(K_row)):
                if K_row[i]==n:
                    ind.append('f'+str(K_col[i])+'_'+str(t))
                    val.append(K_val[i])
            if 'storage' in mode:
                ind.append('s'+str(n)+'_'+str(t))
                val.append(-1.0)
                ind.append('s'+str(n)+'_'+str(t+1))
                val.append(1.0)
            ind.append('b'+str(n)+'_'+str(t))
            ind.append('c'+str(n)+'_'+str(t))
            val.append(-1.0)
            val.append( 1.0)
            var=[]
            for i in ind:
                var.append(network.getVarByName(i))
            network.addConstr(lhs=gb.LinExpr(val,var),sense='=',rhs=float(N[n].mismatch[t]))

    T0+=T
    T=time()-start-T0
    print "Setting constraints took %2.0f:%02.0f" % (T/60.0-np.mod(T/60.0,1),np.mod(T,60))

######################### Setting default values for the solver
###############################################################

    network.setParam("OutputFlag",1)  ## Change this to 1 for debugging individual time steps
    network.setParam("FeasibilityTol",1e-3)
    network.update()

    mean_n=[node.mean for node in N]

###############################################################
###### Initial balancing constraints for capped's case ########
# Weights is just the mean load of a country n divided by the #
# mean load of the sum of the nodes. Global_B is the total    #
# balancing capacity required,  which is distributed by       #
# weights to each node as an upper bound. This part can easily#
# be changed to assign different balancing capacities.        #
###############################################################

    if 'capped' in mode:
        weights = mean_n/sum(mean_n)
        Global_B = np.max( -sum(n.mismatch for n in N) )
        weights *= Global_B*1.001
        for n in N:  ## Restricts non-load nodes to not balance (e.g. Nort-sea wind farms)
            for t in lapse: ## but could have an own flag, of course
                network.getVarByName('b'+str(n.id)+'_'+str(t)).ub=float(weights[n.id])

    network.update()



###############################################################
################# Run the time series  ########################
# The modes 'hybrid' and 'capped' have their own subroutines, #
# all other modes are simply resolved in _solve_flows_. The   #
# status is printed at each time step if verbose. The empty   #
# collectors F, S, B and C are filled as time goes by and only#
# assigned to their resepctive nodes at the end (this is      #
# faster than assigning them to the nodes at each time step)  #
###############################################################


#### Get mismatch & ESS power ####
##################################
#### Add ESS constraints

    if 'storage' in mode:
        for n in N:
            s1 = network.getVarByName('s'+str(n.id)+'_'+str(lapse[0]))
            s2 = network.getVarByName('s'+str(n.id)+'_'+str(lapse[-1]+1))
            network.addConstr(lhs=gb.LinExpr([-1,1],[s1,s2]),sense='=',rhs=float(0))

    #~ s_upper=np.array(N.get_P_discharge(t)) ## NOTE: This may be very wrong!
    #~ s_lower=np.array(-N.get_P_charge(t)) ## NOTE: This may be very wrong!
    #~ for n in range(Nnodes):
        #~ if 'smart' in mode:
            #~ if Delta[n] < 0:
                #~ s_lower[n] = min(s_upper[n],-(N.storage_factor)*Delta[n])
                #~ s_upper[n] = min(s_upper[n],-(N.storage_factor)*Delta[n])
        #~ s_vars[n].ub=float(s_upper[n])
        #~ s_vars[n].lb=float(s_lower[n])

#### Set objective factors based on mode ####
#############################################

    if 'linear' in mode or 'capped' in mode:
        b1 = np.ones(Nnodes)
        c1 = np.zeros(Nnodes)
        b2 = np.zeros(Nnodes)
        c2 = np.zeros(Nnodes)
        if 'storage' in mode:
            c1 = np.ones(Nnodes)

    if 'square' in mode: ## Shares curtailment!
        b1 = np.zeros(Nnodes)
        c1 = np.zeros(Nnodes)
        b2 = np.ones(Nnodes)/mean_n
        c2 = np.ones(Nnodes)/mean_n

##### Define Step-1 objective #####
###################################
    step1_objective=gb.QuadExpr()
    for t in lapse:
        for n in N:
            bvar = network.getVarByName('b'+str(n.id)+'_'+str(t))
            cvar = network.getVarByName('c'+str(n.id)+'_'+str(t))
            step1_objective.addTerms((b1[0],c1[0]),(bvar,cvar))
            step1_objective.addTerms((b2[0],c2[0]),(bvar,cvar),(bvar,cvar))
    network.setObjective(expr=step1_objective, sense=1)

    T=time()-T0-start
    print "Setting step 1 objective took %2.0f:%02.0f" % (T/60.0-np.mod(T/60.0,1),np.mod(T,60))


##### Solve Step-1 objective #########
######################################
    network.update()
    network.optimize()
    v=[]

    T0+=T
    T=time()-T0-start
    print "Solving step 1 took %2.0f:%02.0f" % (T/60.0-np.mod(T/60.0,1),np.mod(T,60))


    BC_opt=network.objVal
    #~ for v in network.getVars():
        #~ print v.varName, v.x, v.ub, v.lb
    #~ for t in lapse:
        #~ v0 = network.getVarByName('s18_'+str(t))
        #~ print v0, v0.x
##### Add constraints for Step-2 #####
######################################
    if 'square' in mode:
        step2_cs = 0
        for t in lapse:
            for n in Nodes:
                b = network.getVarByName('b'+str(n.id)+'_'+str(t))
                c = network.getVarByName('c'+str(n.id)+'_'+str(t))
                b.ub = b.x*1.00001
                b.lb = b.x*0.99999
                c.ub = c.x*1.00001
                c.lb = c.x*0.99999

    else:
        for t in lapse:
            bvars=[]
            for n in N:
                bvars.append(network.getVarByName('b'+str(n.id)+'_'+str(t)))
            RHS = sum(b.x for b in bvars)
            LHS = gb.LinExpr([1]*Nnodes,bvars)
            network.addConstr(lhs=LHS,sense='<',rhs=float(RHS))

    network.update()

    T0+=T
    T=time()-start-T0
    print "Adding step 2 constraints took %2.0f:%02.0f" % (T/60.0-np.mod(T/60.0,1),np.mod(T,60))

###### Set Step-2 Objective ######
##################################

    step2_objective=gb.QuadExpr()
    for t in lapse:
        for l in range(Nlinks):
            fvar = network.getVarByName('f'+str(l)+'_'+str(t))
            step2_objective.addTerms(1.0,fvar,fvar)
    network.setObjective(expr=step2_objective, sense=1)

    T0+=T
    T=time()-start-T0
    print "Setting step 2 objective took %2.0f:%02.0f" % (T/60.0-np.mod(T/60.0,1),np.mod(T,60))

###### Solve Step 2 ######
##########################

    network.update()
    network.optimize()
    #~ for t in lapse:
        #~ v0 = network.getVarByName('s18_'+str(t))
        #~ print v0, v0.x
    #~ v0 = network.getVarByName('s18_'+str(lapse[-1]+1))
    #~ print v0, v0.x

    T0+=T
    T=time()-start-T0
    print "Solving step 2 took %2.0f:%02.0f" % (T/60.0-np.mod(T/60.0,1),np.mod(T,60))

    for t in lapse:
        for l in range(Nlinks):###### save hourly solutions in long-term matrices
            F[l,t]=network.getVarByName('f'+str(l)+'_'+str(t)).x
        for n in range(Nnodes):
            if 'storage' in mode:
                S[n,t]=network.getVarByName('s'+str(n)+'_'+str(t)).x - network.getVarByName('s'+str(n)+'_'+str(t+1)).x
            B[n,t]=network.getVarByName('b'+str(n)+'_'+str(t)).x
            C[n,t]=network.getVarByName('c'+str(n)+'_'+str(t)).x
        for i in N:
            i.balancing[t] = B[i.id,t]
            i.curtailment[t] = C[i.id,t]
            i.storagePower[t] = S[i.id,t]
            i.storageLevel[t] = network.getVarByName('s'+str(n)+'_'+str(t)).x

    for i in N:
        i.storageLevel[lapse[-1]+1] = network.getVarByName('s'+str(n)+'_'+str(lapse[-1]+1)).x
        i._update_()
        i.solved = True

    T=time()-start
    print "Optimisation took %2.0f:%02.0f" % (T/60.0-np.mod(T/60.0,1),np.mod(T,60))
    return dcopy(N),dcopy(F),network


