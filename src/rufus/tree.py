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


def time(g, v):
    '''Determine the time it takes to get from the root to v.

    Arguments:
        g (tl.tree):    the tree
        v (tl.node):    the target vertex

    Returns:
        int, the time needed to get to v
    '''
    if v.is_root():
        return 0

    t = 0
    cur = v
    while not cur.is_root():
        t += cur.data.time()
        cur = g.parent(cur.identifier)

    return t
# end time


def near_capture(g, v, check_capture, dist, v_pursuer, gamma=1.0):
    '''Check what vertices in g are near capture from v.

    Arguments:
        g:              the graph to check
        v:              the vertex to check capture from
        check_capture:  a function that checks if two vertices are near 
                        capture
        dist:           the pursuer distance function
        v_pursuer:      True, if v is a pursuer node

    Note:
        check_capture should be a function with the following signature:
            
            Vertex x Vertex -> bool

        where the first argument is the Vertex of the pursuer and the second
        is the Vertex of the evader. The function should return True is the 
        pursuer can capture the 
    '''
    r = logball(gamma, len(g), v.data.loc.shape[0])

    if v_pursuer:
        _filter = lambda n: (
            (dist(v.data.loc, n.data.loc, v.data.state) < r) and check_capture(v.data, n.data)
        )
    else:
        _filter = lambda n: (
            (dist(n.data.loc, v.data.loc, n.data.state) < r) and check_capture(n.data, v.data)
        )

    return list(g.filter_nodes(_filter))
# end near_capture


def nearest_neighbor(g, z, dist):
    '''Find the vertex in g that is closest to z in real space
  
    Arguments:
        g (tl.tree):    the tree to search
        z (np.ndarray): the point to compare against
        dist (fn):      the distance function

    Returns:
        (vertex, distance)
    '''
    return min(map(lambda n: (n, dist(n.data.loc, z, n.data.state)), g.all_nodes_itr()), key=itemgetter(1))[0]
# end nearest_neighbor


def logball(gamma, n, dim):
    return gamma * (np.log(n) / n)**(1/dim)
# end logball


def near(g, z, dist, gamma=1.0):
    '''Find all vertices in g that are near z.'''
    return within_radius(g, z, logball(gamma, len(g), z.shape[0]), dist)
# end near


def within_radius(g, z, r, dist):
    '''Find all vertices in g that are within radius r of z based on the
    provided distance function.
    '''
    return list(g.filter_nodes(lambda n: dist(n.data.loc, z, n.data.state) < r))
# end near


def remove(g, v):
    '''Remove vertex v and all decendants from g'''
    g.remove_node(v.identifier)
# end remove

