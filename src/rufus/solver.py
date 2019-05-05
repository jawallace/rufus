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
from rufus.analysis import GameSolution
from rufus.game import Actor, Region, Vertex
import rufus.tree as t


class Solver:

    def __init__(self, dt, space, pursuer, evader, check_capture, gamma=1.0):
        '''Constructor.

        Arguments:
            dt:             the time increment
            space:          the GameSpace
            pursuer:        the pursuer Actor 
            evader:         the evader Actor
            check_capture:  a predicate that checks if a pair of vertices
                            (v_p, v_e) are members of the capture set
            gamma:          scaling constant, as described in Karaman et al.
        '''
        self._dt = dt
        self._space = space
        self._pursuer = pursuer
        self._evader = evader
        self._check_capture = check_capture
    # end __init__


    def extend(self, g, z, actor):
        v_nn = t.nearest_neighbor(g, z, actor.distance)
        state, trajectory = actor.steer(v_nn.data.loc, z)

        # TODO if obstacle free
        nearby = t.near(g, z)

        v_min = v_nn
        cost_min = t.time(g, v_min) + len(trajectory)

        for v in nearby:
            candidate_state, candidate_trajectory = actor.steer(v.data.loc, z)
            cost = t.time(g, v) + len(candidate_trajectory)

            if cost < cost_min: # TODO and obstacle free
                v_min = v
                trajectory = candidate_trajectory
                state = candidate_state
                cost_min = cost

        v_new = g.create_node(parent=v_min, data=Vertex(z, state, trajectory))
        t_v_new = t.time(g, v_new)

        for v in nearby:
            if v == v_min:
                continue

            candidate_state, candidate_trajectory = actor.steer(v_new.data.loc, v.data.loc)
            cost = t.time(g, v)
            new_cost = t_v_new + len(candidate_trajectory)
            if t.time(g, v) > new_cost: # TODO and obstacle free
                v.data.trajectory = candidate_trajectory 
                v.data.state = candidate_state
                g.move(v, v_new)

        return v_new, t_v_new
    # end extend


    def solve(self, pursuer_init, evader_init, target, iters=1000):
        # initialization
        g_p = Tree()
        g_p.create_node('origin', data=pursuer_init)

        g_e = Tree()
        g_e.create_node('origin', data=evader_init)

        for _ in range(iters):
            z_e_rand = self._space.sample()
            v_e_new, t_v_e_new = self.extend(g_e, z_e_rand, self._evader) 

            if v_e_new is not None:
                for v_p in t.near_capture(g_p, z_e_new, self._check_capture)
                    if t.time(g_p, v_p) <= t_v_e_new:
                        t.remove(g_e, z_e_new)

            z_p_rand = self._space.sample()
            v_p_new, t_v_p_new = self.extend(g_e, z_p_rand, self._pursuer, self._pursuer)
            if v_p_new is not None:
                for v_e in t.near_capture(g_e, v_p_new, self._check_capture):
                    if t_v_p_new <= t.time(g_e, v_e):
                        t.remove(g_e, v_e)

        return GameSolution(g_e, g_p)
    # end solve

# end Solver

