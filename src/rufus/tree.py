'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games

Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------

This module contains utilities for tree algorithms needed to solve RRT*.
'''

# Standard Imports
from operator import itemgetter

# Third-Party Imports
import numpy as np
import treelib as tl

# Local Imports


def nearest_neighbor(g, z, dist):
    '''Find the vertex in g that is closest to z in real space
  
    Arguments:
        g (tl.tree):    the tree to search
        z (np.ndarray): the point to compare against
        dist (fn):      the distance function

    Returns:
        (vertex, distance)
    '''
    return min(map(lambda n: (n, dist(n.data, z)), g.all_nodes_itr()), key=itemgetter(1))[0]
# end nearest_neighbor


def logball(n):
    vol = gamma * np.log(n) / n
    r = (3 / (4 * np.pi) * vol)^(1/3)
# end logball


def near(g, z, r, dist):
    '''Find all vertices in g that are within radius r of z based on the
    provided distance function.
    '''
    return g.filter_nodes(lambda n: dist(n.data, z) < r)
# end near


def remove(g, v):
    '''Remove vertex v and all decendants from g'''
    g.remove_node(v.identifier)
# end remove

