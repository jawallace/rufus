'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games

Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------

This module contains actors with complex dynamics.
'''

# Standard Imports

# External Imports
import dubins
import numpy as np

# Local Imports
from rufus.game import Actor


class LinearActor(Actor):
    '''A simple actor for test purposes.

    This actor is infinitely maneuverable and moves with a fixed speed.
    '''
    
    def __init__(self, dt, speed):
        '''Constructor.

        Arguments:
            dt:     the time increment
            speed:  the speed of the actor
        '''
        super().__init__(dt)
        assert speed > 0
        self._speed = speed
    # end __init__


    def steer(self, start, end, state):
        direction = end - start
        #distance = np.linalg.norm(end - start)
        distance = self.time(start, end, state)
        time = distance / self._speed
        unit_vector = (direction / distance).reshape((1, -1))

        t = np.arange(0.0, time, self._dt).reshape((-1, 1))

        # lienar actor is stateless, so we just return an empty array
        return np.array([]), start + self._speed * t * unit_vector
    # end steer


    def time(self, start, end, state):
        return np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
        #return np.linalg.norm(end - start)
    # end time

# end LinearActor


class DubinsCar(Actor):
    '''Represents a Dubins Car with unit speed and turning radius w'''

    def __init__(self, dt, w):
        '''Constructor.

        Arguments:
            dt: time increment
            w:  turning radius
        '''
        super().__init__(dt)
        assert w > 0

        self._w = w
    # end __init__


    def steer(self, start, end, state):
        '''State is the orientation of the car'''
        # only supports 2d
        assert start.shape[0] == 2
        assert end.shape[0] == 2

        endstate = 2 * np.pi * np.random.rand() - np.pi

        q0 = (start[0], start[1], state)
        q1 = (end[0], end[1], state)

        path = dubins.shortest_path(q0, q1, self._w)
        cfg, _ = path.sample_many(self._dt)
        cfg = np.array(cfg)

        return cfg[-1, -1], cfg[:, :2]
    # end steer


    def time(self, start, end, state):
        '''We use euclidean distance as a heuristic to improve runtime.

        This method is used to find candidate vertices to merge with,
        so a heuristic is acceptable.
        '''
        return np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    # end time

# end DubinsCar

