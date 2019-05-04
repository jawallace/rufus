'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games

Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------

This module contains the definitions for the core Game types.
'''

# Standard Imports

# External Imports
import numpy as np

# Local Imports


class Actor:
    '''Represents an actor in the game that can interact with the game.

    Implementations of this class should encode the kinematics of each
    participant in the game.
    '''

    def __init__(self, dt):
        '''Constructor.

        Arguments:
            dt: the time increment
        '''
        assert dt > 0
        self._dt = dt
    # end __init__


    def steer(self, start, end):
        '''Determine the optimal trajectory from start to end under the
        kinematics of the Actor.

        Arguments:
            start (np.ndarray): the starting location
            end (np.ndarray):   the ending location

        Returns:
            (state, trajectory)

            state:
                np.ndarray, the actor's state at the final position

            trajectory:
                np.ndarray, the optimal trajectory of the Actor from start to end

        Postcondition:
            a.steer(start, end)[0][0]  == start
            a.steer(start, end)[0][-1] ~ end
        '''
        raise NotImplementedError()
    # end steer


    def time(self, start, end):
        '''Return the minimum time needed to traverse from start to end.

        Arguments:
            start (np.ndarray): the starting location
            end (np.ndarray):   the ending location

        Returns:
            int, the time needed to traverse from start to end
        '''
        return len(self.steer(start, end)[1])
    # end distance

# end Actor


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


    def steer(self, start, end):
        direction = end - start
        distance = np.linalg.norm(end - start)
        time = distance / self._speed
        unit_vector = (direction / distance).reshape((1, -1))

        t = np.arange(0.0, time, self._dt).reshape((-1, 1))

        # lienar actor is stateless, so we just return an empty array
        return np.array([]), start + self._speed * t * unit_vector
    # end steer

# end LinearActor


class GameSpace:
    '''Represents the game space.

    Arguments:
        lower:  the lower bound of each dimension
        upper:  the upper bound of each dimension
    '''


    def __init__(self, lower, upper):
        assert len(lower) == len(upper)
        assert np.all(upper > lower)

        self._lower = lower
        self._upper = upper
        self._range = upper - lower

        self._ndim = len(self._range)
    # end __init__


    def sample(self):
        return self._range * np.random.sample(self._ndim) + self._lower
    # end sample

# end GameSpace


class Vertex:
    '''Represents a location along all possible trajectories of an Actor.

    A Vertex holds data about the state of the Actor at this vertex as well as
    the trajectory that led from the parent Vertex to this Vertex.

    Parent-child relationships are not available from this data structure. This
    structure just aggregates data at a particular state.
    '''

    def __init__(self, loc, state, trajectory):
        # the location of the actor at this vertex 
        self.loc = loc

        # the state of the actor at this vertex
        self.state = state

        # the trajectory from the parent vertex to this vertex
        self.trajectory = trajectory
     # end __init__


    def time(self):
        return len(self.trajectory)
    # end time

# end Vertex

