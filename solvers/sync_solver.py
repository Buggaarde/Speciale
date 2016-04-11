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
def sync_solver(Graph, verbose = 0):
    
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
            Graph[m][n]['flow'] = np.zeros(len(Graph.node[0]['Mismatch']))

    def _step1AddBalMisInjLoadVars(Graph):
        """
        Adds balance, mismatch and injection pattern variables to the gurobi model,
        and names them accordingly.

        Initializes the 'Balance' and 'Mismatch' attributes of the nodes of Graph to be zero.
        
        Parameters
        ----------
        Graph : NetworkX Graph
        """
        for node in Graph.nodes_iter():
            ntwk.addVar(name = 'b' + str(node), lb = -inf, ub = inf) # balance
            ntwk.addVar(name = 'mm' + str(node), lb = -inf, ub = inf) # mismatch
            ntwk.addVar(name = 'ip%d' % node, lb = -inf, ub = inf) # injection pattern
            ntwk.addVar(name = 'l%d' % node, lb = -inf, ub = inf) # load
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
            else: # Update the rhs of the contraints
                mmConstr = ntwk.getConstrByName('mm%d = mismatch in node %d' % (node, node))
                loadConstr = ntwk.getConstrByName('l%d = load in node %d' % (node, node))

                mmConstr.rhs = Graph.node[node]['Mismatch'][step]
                loadConstr.rhs = Graph.node[node]['Load'][step]
        ntwk.update()
        
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

    def _step1GetObj(Graph, step):
        """
        Calculates the objective function to minimize in the step 1 minimization. The
        objective function is  sum balance^2/load  .

        Parameters
        ----------
        Graph : NetworkX Graph

        step : int
            The step in the provided timeseries. If, for example, the timeseries has hourly
            data, step would be the hour currently looked at by the solver.
        """
        balanceSquareSum = gb.QuadExpr()
        for node in Graph.nodes():
            bal = ntwk.getVarByName('b' + str(node))
            load = Graph.node[node]['Load'][step]
            balanceSquareSum.add(bal*bal/load)
        return balanceSquareSum
            
    def _step2AddFlowConstraints(Graph, step = 0):
        """
        Kirchoff's laws. The sum of all flows in or out of a node must equal the injection
        in that node.
        
        If step = 0, adds the above as constraints to the Gurobi model and if step != 0,
        instead update the rhs of the constraints, which is much faster than removing
        the constraint and adding it again with a new rhs.

        Parameters
        ----------
        Graph : NetworkX Graph

        step : int
            The step in the provided timeseries. If, for example, the timeseries has hourly
            data, step would be the hour currently looked at by the solver.
        """
        for node in Graph.nodes_iter():
            ip = ntwk.getVarByName('ip%d' % node).X
            if step == 0: # Add the constraint to the model
                flowSum = gb.LinExpr()                
                for neighbor in Graph.neighbors_iter(node):
                    if node < neighbor:
                        f = ntwk.getVarByName('flow %d->%d' % (node, neighbor))
                        flowSum.add(f)
                    elif neighbor < node:
                        f = ntwk.getVarByName('flow %d->%d' % (neighbor, node))
                        flowSum.add(-f)
                ntwk.addConstr(flowSum == ip, name = 'flows from node %d = ip%d' %(node, node))
            else: # Update the rhs of the contraints
                flowConstr = ntwk.getConstrByName('flows from node %d = ip%d' %(node, node))
                flowConstr.rhs = ip
                
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
            print constr
        print
    
    totalSteps = len(Graph.node[0]['Mismatch'])
    for step in xrange(totalSteps):
        start = timeit.default_timer()
        if step == 0:
            # -- Setting up the model --
            ntwk = gb.Model('Network')
            #ntwk.params.DualReductions = 0 # Makes the solver specify
                                          # whether error is unbond or infeasible
            ntwk.params.OutputFlag = 0    # Tells Gurobi to shut up
            # -- Giving infinity a better name --
            inf = gb.GRB.INFINITY

            # -- Step 1 --
            _step1AddFlowVars(Graph)
            _step1AddBalMisInjLoadVars(Graph)
            _step1UpdateMismatchLoad(Graph, step)
            _step1AddMismatchConstraints(Graph)
            _step1AddInjectionConstraints(Graph)
            step1Obj = _step1GetObj(Graph, step)
            ntwk.setObjective(step1Obj)
            ntwk.optimize()
            _AddBalInjToGraph(Graph)
                
            # -- Step 2 --
            _step2AddFlowConstraints(Graph)
            step2Obj = _step2GetObj(Graph)
            ntwk.setObjective(step2Obj)
            ntwk.optimize()
            _AddFlowsToGraph(Graph, step)
            
            if ntwk.status != gb.GRB.OPTIMAL:
                ntwk.computeIIS()
                ntwk.write('model.ilp')
                print 'Wrote to model.ilp'
        else:
            # -- Step 1 --
            _step1UpdateMismatchLoad(Graph, step)
            ntwk.setObjective(step1Obj)
            ntwk.optimize()
            _AddBalInjToGraph(Graph, step)
            
            # -- Step 2 --
            _step2AddFlowConstraints(Graph, step)
            ntwk.setObjective(step2Obj)
            ntwk.optimize()
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
    sync_solver(ntwk, verbose=1)
