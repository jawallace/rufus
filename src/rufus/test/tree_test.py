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
from rufus.game import Vertex

class TreeTest(unittest.TestCase):

    def setUp(self):
        self._dt = 1.0

        g = Tree()

        self._n0 = g.create_node(
                'a',
                0,
                data=Vertex(
                    np.array([10.0]), 
                    None, 
                    np.array([])
                )
        )
        self._n1 = g.create_node(
                'b',
                1,
                parent=0,
                data=Vertex(
                    np.array([20.0]),
                    None,
                    np.arange(10.0, 20.0, self._dt)
                )
        )
        self._n2 = g.create_node(
                'c',
                2,
                parent=1,
                data=Vertex(
                    np.array([25.0]),
                    None,
                    np.arange(20.0, 25.0, self._dt)
                )
        )
        self._n3 = g.create_node(
                'd',
                3,
                parent=1,
                data=Vertex(
                    np.array([30.0]), 
                    None, 
                    np.arange(20.0, 30.0, self._dt)
                )
        )
        self._n4 = g.create_node(
                'e',
                4,
                parent=0,
                data=Vertex(
                    np.array([5.0]),
                    None,
                    np.arange(10.0, 5.0, -self._dt)
                )
        )

        self.g = g
    # end setUp


    def test_time(self):
        # root node
        self.assertEqual(0, time(self.g, self._n0))

        # one node, positive direction
        self.assertEqual(10, time(self.g, self._n1))

        # one node, negative direction
        self.assertEqual(5, time(self.g, self._n4))

        # two nodes
        self.assertEqual(15, time(self.g, self._n2))
    # end test_time


    def test_nearest_neighbor(self):
        dist = lambda x, y: np.linalg.norm(x - y)

        self.assertEqual(0, nearest_neighbor(self.g, np.array([9.0]), dist).identifier)
        self.assertEqual(1, nearest_neighbor(self.g, np.array([21.0]), dist).identifier)
        self.assertEqual(2, nearest_neighbor(self.g, np.array([23.0]), dist).identifier)
        self.assertEqual(3, nearest_neighbor(self.g, np.array([31.0]), dist).identifier)
        self.assertEqual(4, nearest_neighbor(self.g, np.array([2.0]), dist).identifier)
    # end test_nearest_neighbor


    def test_within_radius(self):
        dist = lambda x, y: np.linalg.norm(x - y)

        nodes = list(within_radius(self.g, np.array([25.0]), 10.0, dist))
        self.assertEqual(3, len(nodes))
        self.assertEqual(set([1, 2, 3]), set([n.identifier for n in nodes]))
    # end test_near


    def test_remove(self):
        n = self.g.get_node(1)
        remove(self.g, n)

        self.assertEqual(2, len(self.g.nodes))
        self.assertEqual(set([0, 4]), set(self.g.nodes.keys()))
    # end test_remove


    def test_logball(self):
        self.assertEqual(0, logball(1.0, 1)) 
        self.assertAlmostEqual(0.38, logball(1.0, 10), places=3)
    # end test_near_capture


    def test_near(self):
        dist = lambda x, y: np.linalg.norm(x - y)

        # note: logball(5000, 5) ~ 7.3
        nodes = near(self.g, 25.0, dist, gamma=5000.0)
        self.assertEqual(3, len(nodes))
        self.assertEqual(set([1, 2, 3]), set([n.identifier for n in nodes]))

        nodes = near(self.g, 0.0, dist, gamma=5000.0)
        self.assertEqual(1, len(nodes))
        self.assertEqual(4, nodes[0].identifier)

        nodes = near(self.g, 50.0, dist, gamma=5000.0)
        self.assertEqual(0, len(nodes))
    # end test_near


    def test_near_capture(self):
        # note: logball(5000, 5) ~ 7.3
        dist = lambda x, y: np.linalg.norm(x - y)

        def _check(v_p, v_e):
            # for the purposes of this test, the capture set requires that 
            # dist(v_p, v_e) < 5 and v_p < v_e
            return 0 < (v_e.loc - v_p.loc) < 5
        # end _check

        ###################################################################
        # test v_pursuer == True case (i.e. self.g are evader nodes)

        # no capture
        t = Tree()
        v_p = t.create_node(data=Vertex(100.0, None, None))
        nodes = near_capture(self.g, v_p, _check, dist, True, gamma=5000.0)
        self.assertEqual(0, len(nodes))

        # one capture - v2
        # note that this tests three cases (using Karaman's notation):
        #   1. Near and CaptureSet (node 2)
        #   2. Near and not CaptureSet (node 1, 3)
        #   3. Not Near (node 0, 4)
        t = Tree()
        v_p = t.create_node(data=Vertex(24.9, None, None))
        nodes = near_capture(self.g, v_p, _check, dist, True, gamma=5000.0)
        self.assertEqual(1, len(nodes))
        self.assertEqual(2, nodes[0].identifier)


        ###################################################################
        # test v_pursuer == False case (i.e. self.g are pursuer nodes)
        
        # no capture
        t = Tree()
        v_e = t.create_node(data=Vertex(100.0, None, None))
        nodes = near_capture(self.g, v_e, _check, dist, False, gamma=5000.0)
        self.assertEqual(0, len(nodes))

        # capture - v2
        # note that this tests three cases (using Karaman's notation):
        #   1. Near and CaptureSet (node 2)
        #   2. Near and not CaptureSet (node 1, 3)
        #   3. Not Near (node 0, 4)
        t = Tree()
        v_e = t.create_node(data=Vertex(25.1, None, None))
        nodes = near_capture(self.g, v_e, _check, dist, False, gamma=5000.0)
        self.assertEqual(1, len(nodes))
        self.assertEqual(2, nodes[0].identifier)
    # end test_near_capture

# end TreeTest

