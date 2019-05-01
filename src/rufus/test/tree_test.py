'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games

Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------

Unit tests for the tree module.
'''

# Standard Imports
import unittest

# Third-Party Imports
import numpy as np
from treelib.tree import Tree

# Local Imports
from rufus.tree import *


class TreeTest(unittest.TestCase):

    def setUp(self):
        g = Tree()

        g.create_node('a', 0, data=np.array([10.0]))
        g.create_node('b', 1, parent=0, data=np.array([20.0]))
        g.create_node('c', 2, parent=1, data=np.array([25.0]))
        g.create_node('d', 3, parent=1, data=np.array([30.0]))
        g.create_node('e', 4, parent=0, data=np.array([5.0]))

        self.g = g
    # end setUp


    def test_nearest_neighbor(self):
        dist = lambda x, y: np.linalg.norm(x - y)

        self.assertEqual(0, nearest_neighbor(self.g, np.array([9.0]), dist).identifier)
        self.assertEqual(1, nearest_neighbor(self.g, np.array([21.0]), dist).identifier)
        self.assertEqual(2, nearest_neighbor(self.g, np.array([23.0]), dist).identifier)
        self.assertEqual(3, nearest_neighbor(self.g, np.array([31.0]), dist).identifier)
        self.assertEqual(4, nearest_neighbor(self.g, np.array([2.0]), dist).identifier)
    # end test_nearest_neighbor


    def test_near(self):
        dist = lambda x, y: np.linalg.norm(x - y)

        nodes = list(near(self.g, np.array([25.0]), 10.0, dist))
        self.assertEqual(3, len(nodes))
        self.assertEqual(set([1, 2, 3]), set([n.identifier for n in nodes]))
    # end test_near


    def test_remove(self):
        n = self.g.get_node(1)
        remove(self.g, n)

        self.assertEqual(2, len(self.g.nodes))
        self.assertEqual(set([0, 4]), set(self.g.nodes.keys()))
    # end test_remove

# end TreeTest

