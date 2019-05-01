'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games

Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------

This module contains the implementation of the solver for the differential game
'''

# Standard Imports

# Third-Party Imports
import numpy as np

# Local Imports


class Actor:

    def __init__(self):
        pass
    # end __init__


    def steer(self, start, end):
        pass
    # end steer


    def distance(self, start, end):
        pass
    # end distance

# end Actor


class GameSpace:

    def __init__(self, lower, upper, target):
        assert len(lower) == len(upper)
        assert np.all(upper > lower)

        self._lower = lower
        self._upper = upper
        self._range = upper - lower
        self._target = target

        self._ndim = len(bounds)
    # end __init__


    def sample(self):
        return self._range * np.random.sample(self._ndim) + self._lower
    # end sample

# end GameSpace


class Solver:

    def __init__(self, space, pursuers, evaders):
        assert len(pursuers) > 0
        assert len(evaders) > 0

        self._space = space
        self._pursuers = pursuers
        self._evaders = evaders
        self._target = target
    # end __init__


    def solve(self, pursuer_init, evader_init, target, iters=1000):
        pass
    # end solve

# end Solver

