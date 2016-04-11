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
def local_solver(Graph, verbose = 0):
    
    #@timing
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
        

        # -- Initializing the 'flow' atribute of the edges to be 0 --
        for (m, n) in Graph.edges_iter():
            Graph[m][n]['flow'] = np.zeros(len(Graph.node[0]['Mismatch']))
        
    #@timing
    def _step1AddBalMisInjLoadVars(Graph):
        """
        Adds backup, curtailment, mismatch and injection pattern variables to the gurobi model, 
        and names them accordingly.
        """
        
        for node in Graph.nodes_iter():
            back = ntwk.addVar(name = 'back' + str(node), lb = -inf, ub = 0) # backup
            c = ntwk.addVar(name = 'c' + str(node), lb = 0, ub = inf) # curtailment
            b = ntwk.addVar(name='b%d' % (node), lb=-inf, ub=inf)
            ntwk.update()
            ntwk.addConstr(b == back + c, name='b%d = back%d + c%d' % (node, node, node))
            # ntwk.addConstr(back <= 0, name='back%d <= 0' % (node))
            # ntwk.addConstr(c >= 0, name='c%d >= 0' % (node))
            ntwk.addVar(name = 'mm' + str(node), lb = -inf, ub = inf) # mismatch
            ntwk.addVar(name = 'ip%d' % node, lb = -inf, ub = inf) # injection pattern
            ntwk.addVar(name = 'l%d' % node, lb = -inf, ub = inf)  # load
        ntwk.update()

        for node in Graph.nodes_iter():
            Graph.node[node]['Balance'] = np.zeros(len(Graph.node[0]['Mismatch']))
            Graph.node[node]['Injection Pattern'] = np.zeros(len(Graph.node[0]['Mismatch']))
        
    # #@timing    
    # def _step1LoadGraphLoads(Graph, step = 0):
    #     """
    #     Loads the 'load' attribute fomr the nodes of Graph into the gurobi model.
    #     """
    #     for node in Graph.nodes():
            
    #         if step == 0:
    #             ntwk.addConstr
        
        
    #@timing    
    def _step1UpdateMismatchLoadOLD(Graph, step = 0):
        """
        Loads the 'mismatch' attributes from the nodes of Graph into the gurobi model.
        """
        
        for node in Graph.nodes():
            mm = ntwk.getVarByName('mm%d' % node)
            l = ntwk.getVarByName('l%d' % node)
            if step == 0:
                ntwk.addConstr(mm == Graph.node[node]['Mismatch'][step],\
                               name = 'mm%d = %2.2f' % (node, Graph.node[node]['Mismatch'][step]))
                ntwk.addConstr(l == Graph.node[node]['Load'][step],\
                               name = 'l%d = %2.2f' % (node, Graph.node[node]['Load'][step]))
            else:
                ntwk.remove(ntwk.getConstrByName('mm%d = %2.2f'\
                                                 % (node, Graph.node[node]['Mismatch'][step - 1])))
                ntwk.addConstr(mm == Graph.node[node]['Mismatch'][step],\
                               name = 'mm%d = %2.2f' % (node, Graph.node[node]['Mismatch'][step]))
                ntwk.remove(ntwk.getConstrByName('l%d = %2.2f'\
                                                 % (node, Graph.node[node]['Load'][step - 1])))
                ntwk.addConstr(l == Graph.node[node]['Load'][step],\
                               name = 'l%d = %2.2f' % (node, Graph.node[node]['Load'][step]))
        ntwk.update()

    #@timing
    def _step1UpdateMismatchLoad(Graph, step = 0):
        """
        If step = 0 this function adds the mismatch and load constraints, which basically
        just tells the model the known values for the mismatch and load.
        If step is any other than the first one, this function only updates the values of the rhs
        of the constraints, which is MUCH faster than removing the constraint and then re-adding 
        it with the new rhs-values.
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
        
    #@timing
    def _step1RemoveMismatchConstraints(Graph, step = 0):
        for node in Graph.nodes_iter():
            ntwk.remove(ntwk.getConstrByName('mm%d = %2.2f' % (node, Graph.node[node]['Mismatch'][step])))
        
    #@timing    
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

    #@timing
    def _step1AddInjectionConstraints(Graph):
        
        injSum = gb.LinExpr()
        for node in Graph.nodes_iter():
            term = ntwk.getVarByName('ip%d' %node)
            injSum.add(term)
        ntwk.addConstr(injSum == 0, name = 'sum of ip\'s = 0')
        ntwk.update()

    #@timing
    def _step1GetObj(Graph, step):
        backupSum = gb.LinExpr()
        for node in Graph.nodes():
            backup = ntwk.getVarByName('back%d' % (node))
            backupSum.add(-backup)
        return backupSum
            
    #@timing
    def _step1SetObjectiveFunction(Graph):
        
        balanceSquareSum = gb.QuadExpr()
        for node in Graph.nodes():
            term = ntwk.getVarByName('b' + str(node))
            balanceSquareSum.add(term*term)
        ntwk.setObjective(balanceSquareSum, sense = gb.GRB.MINIMIZE)
        ntwk.update()
                    
            
    def _step2AddBackupConstraint(Graph, step):
        backupSum = gb.LinExpr()
        backupTargetValue = 0
        for node in Graph.nodes_iter():
            backupSum.add(ntwk.getVarByName('back%d' % node))
            backupTargetValue += Graph.node[node]['Mismatch'][step]
        # print backupTargetValue
            
        if step == 0:
            ntwk.addConstr(backupSum == backupTargetValue,
                           name='step1 objective, sum of backups = sum of mismatches')
        else:
            ntwk.getConstrByName('step1 objective, sum of backups = sum of mismatches').rhs = backupTargetValue
        ntwk.update()
        
    #@timing
    def _step2AddFlowConstraints(Graph, step = 0):
        """
        Kirchoff's laws
        """
        for node in Graph.nodes_iter():
            ip = ntwk.getVarByName('ip%d' % node)
            if step == 0:
                flowSum = gb.LinExpr()                
                for neighbor in Graph.neighbors_iter(node):
                    if node < neighbor:
                        f = ntwk.getVarByName('flow %d->%d' % (node, neighbor))
                        flowSum.add(f)
                    elif neighbor < node:
                        f = ntwk.getVarByName('flow %d->%d' % (neighbor, node))
                        flowSum.add(-f)
                ntwk.addConstr(flowSum == ip, name = 'flows from node %d = ip%d' %(node, node))
            else:
                # flowConstr = ntwk.getConstrByName('flows from node %d = ip%d' %(node, node))
                # flowConstr.rhs = ip
                pass
                    

    #@timing
    def _step1UpdateFlowConstraints(Graph, step = 0):
        for node in Graph.nodes():
            flowConstr = ntwk.getConstrByName('flows from node %d = ip%d' % (node, node))
            flowConstr.rhs = 0


    #@timing
    def _step2GetObj(Graph):
        flowSquareSum = gb.QuadExpr()
        for (m, n) in Graph.edges_iter():
            flow = ntwk.getVarByName('flow ' + str(m) + '->' + str(n))
            flowSquareSum.add(flow*flow)
        return flowSquareSum
            
    #@timing
    def _step2SetObjectiveFunction(Graph):
        
        flowSquareSum = gb.QuadExpr()
        for (m, n) in Graph.edges_iter():
            flow = ntwk.getVarByName('flow ' + str(m) + '->' + str(n))
            flowSquareSum.add(flow*flow)
        ntwk.setObjective(flowSquareSum, sense = gb.GRB.MINIMIZE)
        ntwk.update()
        
    #@timing
    def _AddBalInjToGraph(Graph, step = 0):
        for node in Graph.nodes_iter():
            #print ntwk.getVarByName('ip%d' % node)
            Graph.node[node]['Injection Pattern'][step] = ntwk.getVarByName('ip%d' % node).X
            Graph.node[node]['Mismatch'][step] = ntwk.getVarByName('mm%d' % node).X
            Graph.node[node]['Balance'][step] = Graph.node[node]['Mismatch'][step] - Graph.node[node]['Injection Pattern'][step]
            
    #@timing
    def _AddFlowsToGraph(Graph, step):
        for (m, n) in Graph.edges_iter():
            Graph[m][n]['flow'][step] = ntwk.getVarByName('flow %d->%d' % (m, n)).X

    def _printEverything(message):
        print message
        for var in ntwk.getVars():
            print var
        for constr in ntwk.getConstrs():
            print constr, constr.rhs
        print


    
    #print 'number of mismatches: %d ' % len(Graph.node[0]['Mismatch'])
    totalSteps = len(Graph.node[0]['Mismatch'])
    sstart = timeit.default_timer()
    for step in xrange(totalSteps):
        start = timeit.default_timer()
        #sys.stdout.write('%s\r' % ('Step %d of %d' % (step + 1, totalSteps)))
        #print '--------------------STEP %d of %d--------------------\r' % (step + 1, totalSteps)
        if step == 0:
            # -- Step 1 --
            ntwk = gb.Model('Network')
            #ntwk.params.DualReductions = 0
            ntwk.params.OutputFlag = 0
            # -- Giving infinity a better name --
            inf = gb.GRB.INFINITY

            _step1AddFlowVars(Graph)
            _step1AddBalMisInjLoadVars(Graph)
            
            _step1UpdateMismatchLoad(Graph, step)
            _step1AddMismatchConstraints(Graph)
            
            _step1AddInjectionConstraints(Graph)
            
            
            # step1Obj = _step1GetObj(Graph, step)
            # ntwk.setObjective(step1Obj)
            # _printEverything('')
            # ntwk.optimize()
            # if ntwk.status != gb.GRB.OPTIMAL:
                # print(ntwk.status)
                # ntwk.computeIIS()
                # ntwk.write('model.ilp')
                # print 'Wrote to model.ilp'
            # _printEverything('')
            # _AddBalInjToGraph(Graph)
            
                
            # -- Step 2 --
            _step2AddBackupConstraint(Graph, step) # instead of minimizing the sum of backups
            _step2AddFlowConstraints(Graph)
            step2Obj = _step2GetObj(Graph)
            # _printEverything('')
            ntwk.setObjective(step2Obj)
            ntwk.optimize()
            # _printEverything('')
            _AddBalInjToGraph(Graph, step)
            _AddFlowsToGraph(Graph, step)
            
            if ntwk.status != gb.GRB.OPTIMAL:
                print(ntwk.status)
                ntwk.computeIIS()
                ntwk.write('model.ilp')
                print 'Wrote to model.ilp'
        else:
            # -- Step 1 --
            # _step1UpdateFlowConstraints(Graph, step)
            _step1UpdateMismatchLoad(Graph, step)


            # _printEverything('')
            # ntwk.setObjective(step1Obj)
            # ntwk.optimize()
            # _AddBalInjToGraph(Graph, step)
            
            # -- Step 2 --
            _step2AddBackupConstraint(Graph, step) # instead of minimizing the sum of backups
            # _step2AddFlowConstraints(Graph, step)
            ntwk.setObjective(step2Obj)
            ntwk.optimize()
            _AddBalInjToGraph(Graph, step)
            _AddFlowsToGraph(Graph, step)
            if ntwk.status != gb.GRB.OPTIMAL:
                ntwk.computeIIS()
                ntwk.write('model.ilp')
                print 'Wrote to model.ilp'
        end = timeit.default_timer()
        #sys.stdout.write('time to run step %d of %d: %3.1f sec\r' % ((step + 1), totalSteps, (end - sstart)))
    #print '\nAverage time pr step: %f' % ((end - sstart)/totalSteps)
    

    # for constr in ntwk.getConstrs():
    #     print '%s = %s' % (constr.ConstrName, constr.rhs)
        
    
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
    ntwk = nx.powerlaw_cluster_graph(5, 3, 0.2)
    steps = 5
    for node in ntwk.nodes():
        ntwk.node[node]['Mismatch'] = np.random.randn(steps) - 0.25
        ntwk.node[node]['Load'] = np.ones(steps)

    for step in xrange(steps):
        totMis = sum(ntwk.node[node]['Mismatch'][step] for node in ntwk.nodes())
        if totMis > 0:
            ntwk.node[0]['Mismatch'][step] -= totMis*1.1

    local_solver(ntwk, verbose=1)
