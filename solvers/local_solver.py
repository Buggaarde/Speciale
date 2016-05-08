#! /usr/bin/env python
import numpy as np
import gurobipy as gb
import networkx as nx
import matplotlib.pyplot as plt
import timeit
import sys

def timing(f):
    def timed(*args, **kwargs):
        ts = timeit.default_timer()
        result = f(*args, **kwargs)
        te = timeit.default_timer()

        print 'time to run %r: %f sec' % (f.__name__, (te - ts))
        return result
    return timed

@timing
def local_solver(energy_network, verbose = 0):
    
    Graph = energy_network.copy()      # Copies energy_network to Graph such that
                                       # it is possible to return Graph without
                                       # modifying energy_network, allowing 
                                       # multiple solvers to be used on the
                                       # same network.

    def _step1AddFlowVars(Graph):
        """
        Adding flow variables to the gurobi model. 
        
        Adds flows from node m->n if m<n and the (m, n)'th position in the adjacency matrix
        of Graph is non-zero. Only adds flows from m->n and not n->m because direction
        in this model is shown by a sign difference, but otherwise the Graph is not directed.

        Initializes the 'flow' attributes of the edges of Graph to be zero.
        
        Parameters
        ----------
        Graph : NetworkX Graph
        """
        adjacency = nx.adjacency_matrix(Graph).todense()
        for row in xrange(len(adjacency)):
            for col in xrange(len(adjacency)):
                if adjacency[row, col] != 0:
                    if row < col:
                        ntwk.addVar(name = 'flow %d->%d' % (row, col), lb = -inf, ub = inf)
        ntwk.update()

        # -- Initializing the 'flow' atribute of the edges to be 0 --
        for (m, n) in Graph.edges_iter():
            Graph[m][n]['Flow'] = np.zeros(len(Graph.node[0]['Mismatch']))
        
    def _step1AddBalMisInjLoadVars(Graph):
        """
        Adds backup, balance, curtailment, mismatch and injection pattern variables
        to the gurobi model, and names them accordingly.

        Initializes the 'Balance' and 'Mismatch' attributes of the nodes of Graph to be zero.
        
        Parameters
        ----------
        Graph : NetworkX Graph
        """
        for node in Graph.nodes_iter():
            back = ntwk.addVar(name = 'back' + str(node), lb = -inf, ub = 0) # backup
            c = ntwk.addVar(name = 'c' + str(node), lb = 0, ub = inf) # curtailment
            b = ntwk.addVar(name='b%d' % (node), lb=-inf, ub=inf)
            ntwk.update()
            ntwk.addConstr(b == back + c, name='b%d = back%d + c%d' % (node, node, node))
            ntwk.addVar(name = 'mm' + str(node), lb = -inf, ub = inf) # mismatch
            ntwk.addVar(name = 'ip%d' % node, lb = -inf, ub = inf) # injection pattern
            ntwk.addVar(name = 'l%d' % node, lb = -inf, ub = inf)  # load
        ntwk.update()

        for node in Graph.nodes_iter():
            Graph.node[node]['Balance'] = np.zeros(len(Graph.node[0]['Mismatch']))
            Graph.node[node]['Injection Pattern'] = np.zeros(len(Graph.node[0]['Mismatch']))
        
    def _step1UpdateMismatchLoad(Graph, step = 0):
        """
        Makes the model aware of the provided values for mismatch and load.

        If step = 0 this function adds the mismatch and load constraints, which basically
        just tells the model the provided values for the mismatch and load.

        If step != 0, this function only updates the values of the rhs of the constraints,
        which is MUCH faster than removing the constraint and then re-adding it with
        the new rhs-values.
        
        Parameters
        ----------
        Graph : NetworkX Graph

        step : int
            The step in the provided timeseries. If, for example, the timeseries has hourly
            data, step would be the hour currently looked at by the solver.
        """
        for node in Graph.nodes():
            if step == 0: # Add variables and constraints to the Gurobi model
                mm = ntwk.getVarByName('mm%d' % node)
                l = ntwk.getVarByName('l%d' % node)
                mmConstr = ntwk.addConstr(mm == Graph.node[node]['Mismatch'][step],\
                               name = 'mm%d = mismatch in node %d' % (node, node))
                
                loadConstr = ntwk.addConstr(l == Graph.node[node]['Load'][step],\
                               name = 'l%d = load in node %d' % (node, node))                
            else: # Update the rhs of the constraints
                mmConstr = ntwk.getConstrByName('mm%d = mismatch in node %d' % (node, node))
                loadConstr = ntwk.getConstrByName('l%d = load in node %d' % (node, node))

                mmConstr.rhs = Graph.node[node]['Mismatch'][step]
                loadConstr.rhs = Graph.node[node]['Load'][step]
        ntwk.update()
        
    def _step1RemoveMismatchConstraints(Graph, step = 0):
        for node in Graph.nodes_iter():
            ntwk.remove(ntwk.getConstrByName('mm%d = %2.2f' \
                                             % (node, Graph.node[node]['Mismatch'][step])))
        
    def _step1AddMismatchConstraints(Graph):
        """
        Tells the model that  balance + injection pattern = mismatch  in each node.
        
        Parameters
        ----------
        Graph : NetworkX Graph
        """
        for node in Graph.nodes_iter():
            balance = ntwk.getVarByName('b' + str(node))
            injPat = ntwk.getVarByName('ip%d' % node)            
            mismatch = ntwk.getVarByName('mm' + str(node))            
            ntwk.addConstr(balance + injPat == mismatch, name = balance.VarName + ' + ' +\
                           injPat.VarName + ' = ' + mismatch.VarName)
        ntwk.update()

    def _step1AddInjectionConstraints(Graph):
        """
        Tells the model to have a balanced injection pattern, which is to say
        sum of injection patterns must equal zero.

        Parameters
        ----------
        Graph : NetworkX Graph
        """
        injSum = gb.LinExpr()
        for node in Graph.nodes_iter():
            term = ntwk.getVarByName('ip%d' %node)
            injSum.add(term)
        ntwk.addConstr(injSum == 0, name = 'sum of ip\'s = 0')
        ntwk.update()
            
    def _step2AddBackupConstraint(Graph, step):
        """
        Because the step 1 minimization is unnecessary for the local flowscheme, (since
        the minimum sum of balances is just the sum of injection patterns,) that step is  
        instead replaced by a constraint telling the model that the sum of balances has to
        equal the sum of injection patterns.

        If step = 0 the constraint is added to the model, and if step != 0 the rhs of the
        constraint is instead updated, which is much faster than removing the constraint
        and adding it again with a new rhs.

        Parameters
        ----------
        Graph : NetworkX Graph

        step : int
            The step in the provided timeseries. If, for example, the timeseries has hourly
            data, step would be the hour currently looked at by the solver.
        """
        backupSum = gb.LinExpr()
        backupTargetValue = 0
        for node in Graph.nodes_iter(): # Find the sum of mismatches
            backupSum.add(ntwk.getVarByName('back%d' % node))
            backupTargetValue += Graph.node[node]['Mismatch'][step]
        # If there is a net positive mismatch in the system, the constraint
        # instead becomes 0.
        if backupTargetValue > 0:
            backupTargetValue = 0
            
        if step == 0: # Add constraint to the Gurobi model
            ntwk.addConstr(backupSum == backupTargetValue,
                           name='step1 objective, sum of backups = sum of mismatches')
        else: # Update the rhs of the constraint
            ntwk.getConstrByName('step1 objective, sum of backups = sum of mismatches').rhs\
                = backupTargetValue
        ntwk.update()
        
    def _step2AddFlowConstraints(Graph, step = 0):
        """
        Kirchoff's laws. The sum of all flows in or out of a node must equal the injection
        in that node.
        
        Adds the above as constraints to the Gurobi model.

        Parameters
        ----------
        Graph : NetworkX Graph

        step : int
            The step in the provided timeseries. If, for example, the timeseries has hourly
            data, step would be the hour currently looked at by the solver.
        """
        for node in Graph.nodes_iter():
            ip = ntwk.getVarByName('ip%d' % node) # The injection pattern
            flowSum = gb.LinExpr()                # The sum of flows in a node
            for neighbor in Graph.neighbors_iter(node):
                if node < neighbor:
                    f = ntwk.getVarByName('flow %d->%d' % (node, neighbor))
                    flowSum.add(f)
                elif neighbor < node:
                    f = ntwk.getVarByName('flow %d->%d' % (neighbor, node))
                    flowSum.add(-f)
            ntwk.addConstr(flowSum == ip, name = 'flows from node %d = ip%d' %(node, node))

    def _step2GetObj(Graph):
        """
        Tells the model to minimize the sum of square flows. 
        For unconstrained flows, this is the same as finding the flows in the DC approximation.
        
        The objective is calculated only once, in order to be more efficient.

        Parameters
        ----------
        Graph : NetworkX Graph
        """
        flowSquareSum = gb.QuadExpr()
        for (m, n) in Graph.edges_iter():
            flow = ntwk.getVarByName('flow %d->%d' % (m, n))
            flowSquareSum.add(flow*flow)
        return flowSquareSum
            
    def _AddBalInjToGraph(Graph, step = 0):
        """
        Adds the calculated injection pattern and balances to Graph as node attributes.

        Parameters
        ----------
        Graph : NetworkX Graph

        step : int
            The step in the provided timeseries. If, for example, the timeseries has hourly
            data, step would be the hour currently looked at by the solver.
        """
        for node in Graph.nodes_iter():
            Graph.node[node]['Injection Pattern'][step] = ntwk.getVarByName('ip%d' % node).X
            Graph.node[node]['Mismatch'][step] = ntwk.getVarByName('mm%d' % node).X
            Graph.node[node]['Balance'][step] = Graph.node[node]['Mismatch'][step] \
                                                - Graph.node[node]['Injection Pattern'][step]
            
    def _AddFlowsToGraph(Graph, step):
        """
        Adds the calculated flows to Graph as edge attributes.

        Parameters
        ----------
        Graph : NetworkX Graph

        step : int
            The step in the provided timeseries. If, for example, the timeseries has hourly
            data, step would be the hour currently looked at by the solver.
        """
        for (m, n) in Graph.edges_iter():
            Graph[m][n]['flow'][step] = ntwk.getVarByName('flow %d->%d' % (m, n)).X

    def _printEverything(message):
        """
        A helper function useful in debugging circumstances.

        Parameters
        ----------
        message: string
            The string you want to be displayed above all the variables and constraints
            in order to be able to tell apart all the information.
        """
        print message
        for var in ntwk.getVars():
            print var
        for constr in ntwk.getConstrs():
            print constr, constr.rhs
        print

    totalSteps = len(Graph.node[0]['Mismatch'])
    for step in xrange(totalSteps):
        if step == 0:  # This distinction is made because adding
                       # variables to the gurobi model is expensive
                       # and therefore we only want to do it for
                       # the first time step.
            # -- Setting up the gurobi model --
            ntwk = gb.Model('Network')
            # ntwk.params.DualReductions = 0 # Makes the solver specifiy
                                           # whether error is unbound or infeasible
            ntwk.params.OutputFlag = 0     # Tells Gurobi to shut up 
            # -- Giving infinity a better name --
            inf = gb.GRB.INFINITY # This is used inside the helper-functions

            # -- Step 1 --
            _step1AddFlowVars(Graph)
            _step1AddBalMisInjLoadVars(Graph)
            _step1UpdateMismatchLoad(Graph, step)
            _step1AddMismatchConstraints(Graph)
            _step1AddInjectionConstraints(Graph)
                
            # -- Step 2 --
            _step2AddBackupConstraint(Graph, step) # instead of minimizing the sum of backups
            _step2AddFlowConstraints(Graph, step)
            step2Obj = _step2GetObj(Graph)
            ntwk.setObjective(step2Obj)
            ntwk.optimize()
            _AddBalInjToGraph(Graph, step)
            _AddFlowsToGraph(Graph, step)
            
            if ntwk.status != gb.GRB.OPTIMAL:
                print(ntwk.status)
                ntwk.computeIIS()
                ntwk.write('model.ilp')
                print 'Wrote to model.ilp'
        else: # The first step is over, so we can instead just update
              # all the values, instead of removing and re-adding
              # gurobi constraints.
            # -- Step 1 --
            _step1UpdateMismatchLoad(Graph, step)
            
            # -- Step 2 --
            _step2AddBackupConstraint(Graph, step) # instead of minimizing the sum of backups
            ntwk.setObjective(step2Obj)
            ntwk.optimize()
            _AddBalInjToGraph(Graph, step)
            _AddFlowsToGraph(Graph, step)
            if ntwk.status != gb.GRB.OPTIMAL:
                ntwk.computeIIS()
                ntwk.write('model.ilp')
                print 'Wrote to model.ilp'
    
    if verbose == 1:
        for step in xrange(len(Graph.node[0]['Mismatch'])):
            print '-------------STEP %d-------------' % step
            for node in Graph.nodes():
                print 'NODE %d' % node
                print 'mismatch: ' + str(Graph.node[node]['Mismatch'][step])
                print 'balance: ' + str(Graph.node[node]['Balance'][step])
                print 'injection pattern: ' + str(Graph.node[node]['Injection Pattern'][step])
                
            print
            for(m, n) in Graph.edges_iter():
                print 'flow %d->%d: ' % (m, n) + str(Graph[m][n]['flow'][step])      

    return Graph

if __name__ == '__main__':
    """
    The mismatches are randomly chosen from a normal distribution, in the example below.
    
    The loads in this basic example are chosen to be 1 in order to quickly be able to
    verify the results.
    """
    ntwk = nx.powerlaw_cluster_graph(5, 3, 0.2)
    steps = 5
    for node in ntwk.nodes():
        ntwk.node[node]['Mismatch'] = np.random.randn(steps)
        ntwk.node[node]['Load'] = np.ones(steps)

    solved_ntwk = local_solver(ntwk, verbose=1)
