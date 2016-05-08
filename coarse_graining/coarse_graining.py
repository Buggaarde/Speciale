#! /usr/bin/env python3

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def merge_nodes(G, nodes, new_node, attr_dict=None, **attr):
    """
    Merges the selected `nodes` of the graph G into one `new_node`,
    meaning that all the edges that pointed to or from one of these
    `nodes` will point to or from the `new_node`.
    attr_dict and **attr are defined as in `G.add_node`.
    """
    
    G.add_node(new_node, attr_dict, **attr) # Add the 'merged' node
    
    for n1,n2,data in G.edges(data=True):
        # For all edges related to one of the nodes to merge,
        # make an edge going to or coming from the `new gene`.
        if n1 in nodes:
            G.add_edge(new_node,n2,data)
        elif n2 in nodes:
            G.add_edge(n1,new_node,data)
    
    for n in nodes: # remove the merged nodes
        G.remove_node(n)

def coarse_graining(Network, x_split=1, y_split=1, scale=1):
    ''' 
    Assumes a network with the positions of the nodes in the range
    x=[0, 1], y=[0, 1].
    If scale is different from 1, then the ranges are assumed to be
    multiplied by that scale.
    x_split and y_split indicate how many sections in either horizontal
    or vertical directions is desired. The sections are evenly 
    distributed along the axes.
    '''
    x_lims = scale*np.linspace(-0.01, 1.01, x_split + 1)
    y_lims = scale*np.linspace(-0.01, 1.01, y_split + 1)
    index_node_dict = {}
    # Build the index_node_dict
    for node in Network.nodes_iter():
        pos = Network.node[node]['pos']
        x_pos = pos[0]
        y_pos = pos[1]
        # give the nodes an identifier telling us where in the new grid the
        # node is.
        x_index = np.where(x_lims > x_pos)[0][0]
        y_index = np.where(y_lims > y_pos)[0][0]
        if (x_index, y_index) in index_node_dict:
            index_node_dict[(x_index, y_index)].append(node)
        else:
            index_node_dict[(x_index, y_index)] = [node]

    total_nodes = len(Network.nodes())
    # We need names for the new, merged notes
    new_nodes = list(range(total_nodes, total_nodes + len(index_node_dict)))
    Coarse_network = Network.copy() # Make copy of old network to execute
                                    # merging procedure on

    for index, nodes in index_node_dict.items():
        # calculate position of new, merged node
        x_pos_sum = 0
        y_pos_sum = 0
        for node in nodes:
            pos = Network.node[node]['pos']
            x_pos_sum += pos[0]
            y_pos_sum += pos[1]
        # The new position is the average of the old
        new_pos = (x_pos_sum/len(nodes), y_pos_sum/len(nodes))
        new_node = new_nodes.pop()
        merge_nodes(Coarse_network, nodes, new_node, pos=new_pos)

    return Coarse_network


if __name__ == '__main__':
    

    P = nx.powerlaw_cluster_graph(50, 3, 0.2)
    # pos = nx.get_node_attributes(P, 'pos')
    pos = nx.spectral_layout(P)
    nx.set_node_attributes(P, 'pos', pos)
    
    pos0 = nx.get_node_attributes(P, 'pos')

    Coarse_P = coarse_graining(P, x_split=4, y_split=4)
    pos1 = nx.get_node_attributes(Coarse_P, 'pos')

    nx.draw_networkx_nodes(P, pos0, alpha=0.15)
    nx.draw_networkx_edges(P, pos0, alpha=0.15)
    nx.draw_networkx_nodes(Coarse_P, pos1, node_size=80, node_color='g')
    nx.draw_networkx_edges(Coarse_P, pos1, width=0.15)
    plt.axis('off')
    plt.savefig('coarsening.pdf', bbox_tight=True)
    plt.clf()
    plt.close()
