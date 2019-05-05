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

# External Imports
import numpy as np

# Local Imports
from rufus.game import BoxRegion, LinearActor


class GameTest(unittest.TestCase):

    def test_linear_actor(self):
        dt = 0.1
        actor = LinearActor(0.1, 10.0)
       
        # 1d
        state, trajectory = actor.steer(0.0, 100.0)
        self.assertEqual(0, state.size)
        self.assertEqual(100, len(trajectory))
        self.assertEqual(0.0, trajectory[0])
        self.assertTrue(np.linalg.norm(100.0 - trajectory[-1]) <= (10.0 * dt))
        self.assertEqual(actor.time(0.0, 100.0), 100)

        # 2d
        state, trajectory = actor.steer(np.array([0.0, 0.0]), np.array([100.0, 100.0]))
        self.assertEqual(0, state.size)
        self.assertEqual(142, len(trajectory))
        np.testing.assert_array_equal(np.array([0.0, 0.0]), trajectory[0])
        self.assertTrue(np.linalg.norm(np.array([100.0, 100.0]) - trajectory[-1]) <= (10.0 * dt))
        self.assertEqual(142, actor.time(np.array([0.0, 0.0]), np.array([100.0, 100.0])))
    # end test_linear_actor


    def test_2d_region(self):
        lower = np.array([0.0, 0.0])
        upper = np.array([100.0, 100.0])
        gspace = BoxRegion(lower, upper)

        samples = []
        for _ in range(1000):
            s = gspace.sample()

            # verify samples fall within the game space
            self.assertTrue(s in gspace)

            samples.append(s)

        samples = np.array(samples)
        # verify the samples cover the game space

        min_seen = np.min(samples, axis=0)
        max_seen = np.max(samples, axis=0)

        vol = np.product(max_seen - min_seen)
        expected_vol = np.product(upper - lower)

        self.assertTrue(vol >= 0.9 * expected_vol)
    # end test_2d_region
   

    def test_3d_region(self):
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([100.0, 100.0, 100.0])
        gspace = BoxRegion(lower, upper)

        samples = []
        for _ in range(1000):
            s = gspace.sample()

            # verify samples fall within the game space
            self.assertTrue(s in gspace)
            samples.append(s)

        samples = np.array(samples)
        # verify the samples cover the game space

        min_seen = np.min(samples, axis=0)
        max_seen = np.max(samples, axis=0)

        vol = np.product(max_seen - min_seen)
        expected_vol = np.product(upper - lower)

        self.assertTrue(vol >= 0.9 * expected_vol)
    # end test_3d_region

# end GameTest


