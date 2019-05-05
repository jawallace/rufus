'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games

Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------

Unit tests for the analysis module.
'''

# Standard Imports
import unittest

# External Imports
import numpy as np
from treelib.tree import Tree

# Local Imports
from rufus.analysis import GameSolution
from rufus.game import Vertex, Region, BoxRegion


class TestGameSolution(unittest.TestCase):


    def setUp(self):
        # contains leaf nodes 3 (pass through) and 8 (endpoint)
        self._target1 = BoxRegion(np.array([50.0, 50.0]), np.array([60.0, 60.0]))

        # contains no nodes
        self._target2 = BoxRegion(np.array([20.0, 80.0]), np.array([50.0, 100.0]))

        # contains non-leaf node 7 (endpoint)
        self._target3 = BoxRegion(np.array([30.0, 10.0]), np.array([60.0, 30.0]))
        
        self._g = Tree()
      
        self._n0 = self._g.create_node(
                identifier=0,
                data=Vertex(np.array([0.0, 0.0]), None, np.array([]))
        )

        self._n1 = self._g.create_node(
                identifier=1,
                parent=self._n0.identifier,
                data=Vertex(
                    np.array([15.0, 15.0]),
                    None,
                    np.array([
                        [0.0, 0.0],
                        [5.0, 5.0],
                        [10.0, 10.0]
                    ])
                )
        )

        self._n2 = self._g.create_node(
                identifier=2,
                parent=self._n1.identifier,
                data=Vertex(
                    np.array([30.0, 45.0]),
                    None,
                    np.array([
                        [15.0, 15.0],
                        [20.0, 25.0],
                        [25.0, 35.0]
                    ])
                )
        )

        self._n3 = self._g.create_node(
                identifier=3,
                parent=self._n2.identifier,
                data=Vertex(
                    np.array([66.0, 60.0]),
                    None,
                    np.array([
                        [30.0, 45.0],
                        [42.0, 50.0],
                        [54.0, 55.0]
                    ])
                )
        )

        self._n4 = self._g.create_node(
                identifier=4,
                parent=self._n3.identifier,
                data=Vertex(
                    np.array([60.0, 90.0]),
                    None,
                    np.array([
                        [66.0, 60.0],
                        [64.0, 70.0],
                        [62.0, 80.0]
                    ])
                )
        )

        self._n5 = self._g.create_node(
                identifier=5,
                parent=self._n3.identifier,
                data=Vertex(
                    np.array([90.0, 60.0]),
                    None,
                    np.array([
                        [66.0, 60.0],
                        [78.0, 60.0]
                    ])
                )
        )

        self._n6 = self._g.create_node(
                identifier=6,
                parent=self._n2.identifier,
                data=Vertex(
                    np.array([20., 75.]),
                    None,
                    np.array([
                        [30.0, 45.0],
                        [25.0, 65.0]
                    ])
                )
        )

        self._n7 = self._g.create_node(
                identifier=7,
                parent=self._n1.identifier,
                data=Vertex(
                    np.array([55.0, 19.0]),
                    None,
                    np.array([
                        [15.0, 15.0],
                        [25.0, 16.0],
                        [35.0, 17.0],
                        [45.0, 18.0]
                    ])
                )
        )

        self._n8 = self._g.create_node(
                identifier=8,
                parent=self._n7.identifier,
                data=Vertex(
                    np.array([55.0, 54.0]),
                    None,
                    np.array([
                        [55.0, 19.0],
                        [55.0, 24.0],
                        [55.0, 29.0],
                        [55.0, 34.0],
                        [55.0, 39.0],
                        [55.0, 44.0],
                        [55.0, 49.0]
                    ]) 
                )
        )

        self._n9 = self._g.create_node(
                identifier=9,
                parent=self._n7.identifier,
                data=Vertex(
                    np.array([75.0, 39.0]),
                    None,
                    np.array([
                        [55.0, 19.0],
                        [65.0, 29.0]
                    ])
                )
        )

        self._path_0178_trajectory = np.array([
            [0.0, 0.0],
            [5.0, 5.0],
            [10.0, 10.0],
            [15.0, 15.0],
            [25.0, 16.0],
            [35.0, 17.0],
            [45.0, 18.0],
            [55.0, 19.0],
            [55.0, 24.0],
            [55.0, 29.0],
            [55.0, 34.0],
            [55.0, 39.0],
            [55.0, 44.0],
            [55.0, 49.0],
            [55.0, 54.0]
        ])

        self._path_0123_trajectory = np.array([
            [0.0, 0.0],
            [5.0, 5.0],
            [10.0, 10.0],
            [15.0, 15.0],
            [20.0, 25.0],
            [25.0, 35.0],
            [30.0, 45.0],
            [42.0, 50.0],
            [54.0, 55.0],
            [66.0, 60.0]
        ])

        self._path_017_trajectory = np.array([
            [0.0, 0.0],
            [5.0, 5.0],
            [10.0, 10.0],
            [15.0, 15.0],
            [25.0, 16.0],
            [35.0, 17.0],
            [45.0, 18.0],
            [55.0, 19.0]
        ])

        # not real, but that's okay for these tests because they only examine
        # the evader path
        #
        # in addition, note that the trajectories allow for varying velocity
        self._soln = GameSolution(self._g, self._g)
    # end setUp


    def test_can_reach(self):
        self.assertTrue(self._soln.can_reach(self._target1))
        self.assertFalse(self._soln.can_reach(self._target2))
        self.assertTrue(self._soln.can_reach(self._target3))
    # end test_can_reach


    def test_all_trajectories_to_target(self):
        results = self._soln.all_trajectories_to_target(self._target1)
        self.assertEqual(2, len(results))

        # sort results by the y-coordinate of the endpoint b/c the order is
        # undetermined
        results = sorted(results, key=lambda p_t: p_t[0][-1].loc[-1])

        # first result should be 0 -> 1 -> 7 -> 8
        path, trajectory = results[0]
        self.assertEqual(4, len(path))
        np.testing.assert_array_equal(path[0].loc, self._n0.data.loc)
        np.testing.assert_array_equal(path[1].loc, self._n1.data.loc)
        np.testing.assert_array_equal(path[2].loc, self._n7.data.loc)
        np.testing.assert_array_equal(path[3].loc, self._n8.data.loc)

        np.testing.assert_array_equal(trajectory, self._path_0178_trajectory)

        # second result should be 0 -> 1 -> 2 -> 3
        path, trajectory = results[1]
        self.assertEqual(4, len(path))
        np.testing.assert_array_equal(path[0].loc, self._n0.data.loc)
        np.testing.assert_array_equal(path[1].loc, self._n1.data.loc)
        np.testing.assert_array_equal(path[2].loc, self._n2.data.loc)
        np.testing.assert_array_equal(path[3].loc, self._n3.data.loc)

        np.testing.assert_array_equal(trajectory, self._path_0123_trajectory)

        results = self._soln.all_trajectories_to_target(self._target2)
        self.assertEqual(0, len(results))

        results = self._soln.all_trajectories_to_target(self._target3)
        self.assertEqual(1, len(results))

        # path is 0 -> 1 -> 7
        path, trajectory = results[0]
        self.assertEqual(3, len(path))
        np.testing.assert_array_equal(path[0].loc, self._n0.data.loc)
        np.testing.assert_array_equal(path[1].loc, self._n1.data.loc)
        np.testing.assert_array_equal(path[2].loc, self._n7.data.loc)

        np.testing.assert_array_equal(trajectory, self._path_017_trajectory)
    # end test_all_trajectories_to_target


    def test_min_trajectory_to_target(self):
        path, trajectory = self._soln.min_trajectory_to_target(self._target1)
        self.assertIsNotNone(path)
        self.assertIsNotNone(trajectory)
        self.assertEqual(4, len(path))
        np.testing.assert_array_equal(path[0].loc, self._n0.data.loc)
        np.testing.assert_array_equal(path[1].loc, self._n1.data.loc)
        np.testing.assert_array_equal(path[2].loc, self._n2.data.loc)
        np.testing.assert_array_equal(path[3].loc, self._n3.data.loc)
        np.testing.assert_array_equal(trajectory, self._path_0123_trajectory)

        path, trajectory = self._soln.min_trajectory_to_target(self._target2)
        self.assertIsNone(path)
        self.assertIsNone(trajectory)

        path, trajectory = self._soln.min_trajectory_to_target(self._target3)
        self.assertIsNotNone(path)
        self.assertIsNotNone(trajectory)
        self.assertEqual(3, len(path))
        np.testing.assert_array_equal(path[0].loc, self._n0.data.loc)
        np.testing.assert_array_equal(path[1].loc, self._n1.data.loc)
        np.testing.assert_array_equal(path[2].loc, self._n7.data.loc)
        np.testing.assert_array_equal(trajectory, self._path_017_trajectory)
    # end test_min_trajectory_to_target


    def test_max_time_trajectory(self):
        path, trajectory = self._soln.max_time_trajectory()
        self.assertEqual(4, len(path))
        np.testing.assert_array_equal(path[0].loc, self._n0.data.loc)
        np.testing.assert_array_equal(path[1].loc, self._n1.data.loc)
        np.testing.assert_array_equal(path[2].loc, self._n7.data.loc)
        np.testing.assert_array_equal(path[3].loc, self._n8.data.loc)
        np.testing.assert_array_equal(trajectory, self._path_0178_trajectory)
    # end test_max_time_trajectory

# end TestGameSolution
