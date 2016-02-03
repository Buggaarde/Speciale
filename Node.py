#!/usr/bin/env python3

class Node:
    
    def __init__(self, xCoord, yCoord):
        self.xCoord = xCoord
        self.yCoord = yCoord
        self.neighbours = []
        self.index = -1
        

    def setNeighbour(self, neighbourNr):
        self.neighbours.append(neighbourNr)

    def setIndex(self, index):
        self.index = index
