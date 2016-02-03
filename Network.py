#!/usr/bin/env python3

import Graph as g

SquareGrid = g.Graph(nrOfNodes = 3**2)
SquareGrid.buildGraph(ofType = "square grid", accordingToRule = "square grid")
#SquareGrid.printAdjecencyMatrix()

SquareGrid.createAdjecencyMatrix()
SquareGrid.printAdjecencyMatrix()



