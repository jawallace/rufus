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


    def steer(self, start, end, state):
        '''Determine the optimal trajectory from start to end under the
        kinematics of the Actor.

        Arguments:
            start:  the starting location
            end:    the ending location
            state:  the initial state

        Returns:
            (state, trajectory)

            state:
                the actor's state at the final position

            trajectory:
                the optimal trajectory of the Actor from start to end

        Postcondition:
            a.steer(start, end)[0][0]  == start
            a.steer(start, end)[0][-1] ~ end
        '''
        raise NotImplementedError()
    # end steer


    def time(self, start, end, state):
        '''Return the minimum time needed to traverse from start to end.

        Arguments:
            start (np.ndarray): the starting location
            end (np.ndarray):   the ending location

        Returns:
            int, the time needed to traverse from start to end
        '''
        return len(self.steer(start, end, state)[1])
    # end distance

# end Actor


class Region:
    '''Represents a region in game space.

    Concrete implementations of this class must override the 
    check_containment and sample methods.

    A point can be checked if it belongs to the region as follows:

        r = SomeConcreteRegion(...) 

        pt = np.array([10.0, 14.0, 11.0])
        if pt in region:
            ...

    '''

    def __contains__(self, pt):
        '''Supports the python in keyword'''
        return self.check_containment(pt)
    # end __contains__


    def check_containment(self, pt):
        '''Check if pt is within the region.

        Arguments:
            pt: the value to check

        Returns:
            True, if pt belongs to the region
        '''
        raise NotImplementedError()
    # end check_containment


    def sample(self):
        '''Sample a value from the region.'''
        raise NotImplementedError()
    # end sample

# end Region


class BoxRegion(Region):
    '''Represents an region that can be described as by an n-orthotope.'''
    
    def __init__(self, lower, upper):
        '''Constructor.

        Arguments:
            lower: the lower bound
            upper: the upper bound
        '''
        assert len(lower) == len(upper)
        assert np.all(upper > lower)

        self.lower = lower
        self.upper = upper
        self._range = upper - lower

        self.ndim = len(self._range)
    # end __init__


    def check_containment(self, pt):
        return np.all(self.lower <= pt) and np.all(pt < self.upper)
    # end check_containment


    def sample(self):
        return self._range * np.random.sample(self.ndim) + self.lower
    # end sample

# end BoxRegion


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

