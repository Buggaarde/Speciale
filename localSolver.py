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

def localSolver(Graph):

    def _step1AddFlowVars(Graph):
        """
        Adding flow variables to the gurobi model. Adds flows from node m->n if m<n and the
        (m, n)'th position in the adjacency matrix of Graph is non-zero.

        Only adds flows from m->n and not n->m because direction in this model is shown by a 
        sign difference, but otherwise the Graph is not directed.

        Initializes the 'flow' attributes of the edges of Graph to be zero.
        """
        
        # -- Adding flow variables to the model --
        # for node in Graph.nodes_iter():
        #     for (m, n) in Graph.edges_iter():
        #         if m < n:
        #             ntwk.addVar(name = 'flow %d->%d' % (m, n), lb = -inf, ub = inf)
        
        adjacency = nx.adjacency_matrix(Graph).todense()
        #print adjacency
        for row in xrange(len(adjacency)):
            for col in xrange(len(adjacency)):
                if adjacency[row, col] != 0:
                    if row < col:
                        ntwk.addVar(name = 'flow %d->%d' % (row, col), lb = -inf, ub = inf)
        ntwk.update()

    def _step1AddBalMisInjLoadVars(Graph):
        """
        Adds balance, mismatch and injection pattern variables to the gurobi model, and names them
        accordingly.
        """
        for node in Graph.nodes_iter():
            ntwk.addVar(name = 'b' + str(node), lb = -inf, ub = inf)
            ntwk.addVar(name = 'mm' + str(node), lb = -inf, ub = inf)
            ntwk.addVar(name = 'ip%d' % node, lb = -inf, ub = inf)
            ntwk.addVar(name = 'l%d' % node, lb = -inf, ub = inf)
        ntwk.update()

    def _step1UpdateMismatchLoad(Graph, step = 0):
        """
        If step = 0 this function adds the mismatch and load constraints, which basically
        just tells the model the known values for the mismatch and load.

        If step is any other than the first one, this function only updates the values of the rhs
        of the constraints, which is MUCH faster than removing the constraint and then re-adding 
        it with the new rhs-value.
        """
        
        for node in Graph.nodes():
            if step == 0:
                mm = ntwk.getVarByName('mm%d' % node)
                l = ntwk.getVarByName('l%d' % node)
                mmConstr = ntwk.addConstr(mm == Graph.node[node]['Mismatch'][step],\
                               name = 'mm%d = mismatch in node %d' % (node, node))
                
                loadConstr = ntwk.addConstr(l == Graph.node[node]['Load'][step],\
                               name = 'l%d = load in node %d' % (node, node))                
            else:
                mmConstr = ntwk.getConstrByName('mm%d = mismatch in node %d' % (node, node))
                loadConstr = ntwk.getConstrByName('l%d = load in node %d' % (node, node))

                mmConstr.rhs = Graph.node[node]['Mismatch'][step]
                loadConstr.rhs = Graph.node[node]['Load'][step]

        ntwk.update()

    def _step1AddMismatchConstraints(Graph):
        """
        Tells the model that  balance + injection pattern = mismatch  in each node.
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
        Tells the model that the sum of the injection pattern must equal zero.
        """
        injNames = []
        injSum = gb.LinExpr()
        for node in Graph.nodes_iter():
            term = ntwk.getVarByName('ip%d' %node)
            injSum.add(term)
            injNames.append('ip%d' % node)
        ntwk.addConstr(injSum == 0, name = 'sum of ip\'s = 0')
        ntwk.update()

    def _step1GetObj(Graph, step):
        balanceSquareSum = gb.QuadExpr()
        for node in Graph.nodes():
            bal = ntwk.getVarByName('b' + str(node))
            load = Graph.node[node]['Load'][step]
            balanceSquareSum.add(bal*bal/load)
        return balanceSquareSum


    def _AddBalInjToGraph(Graph, step = 0):
        for node in Graph.nodes_iter():
            #print ntwk.getVarByName('ip%d' % node)
            Graph.node[node]['Injection Pattern'][step] = ntwk.getVarByName('ip%d' % node).X
            Graph.node[node]['Mismatch'][step] = ntwk.getVarByName('mm%d' % node).X
            Graph.node[node]['Balance'][step] = Graph.node[node]['Mismatch'][step] - Graph.node[node]['Injection Pattern'][step]

            
    totalSteps = len(Graph.node[0]['Mismatch'])
    for step in xrange(totalSteps):
        if step == 0:
            ntwk = gb.Model('Network')
            ntwk.params.OutputFlag = 0
            # -- Giving infinity a better name --
            inf = gb.GRB.INFINITY

            # -- Step 1 --
            _step1AddFlowVars(Graph)
            _step1AddBalMisInjLoadVars(Graph)
            
            _step1UpdateMismatchLoad(Graph, step)
            _step1AddMismatchConstraints(Graph)
            
            _step1AddInjectionConstraints(Graph)
            step1Obj = _step1GetObj(Graph, step) # Only calculating this in step 0 saves computation
            ntwk.setObjective(step1Obj)
            
            ntwk.optimize()
            
            _AddBalInjToGraph(Graph)

            # -- Step 2 --
        else:
            pass

if __name__ == '__main__':
    pass    
