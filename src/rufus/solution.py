'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games

Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------

This module contains the logic to solve differential games.
'''

# Standard Imports

# Third Party Imports
import sympy

# Local Imports
from rufus.game import *


def _compute_value_rpe(game, payoff):
    '''Compute the RPE of the value function (Vdot_k).

    The RPEs needed to solve the differential game consist of two sets:
        (1) the RPE of each component of the n-dimensional value function 
            (where n = game.cardinality())
        (2) the RPE of each componenet of the n-dimensional game space 
            (where n = game.cardinality())

    This function computes (1) [Isaacs, Eqn 4.5.3]

    Arguments:
        game:       the game to process
        payoff:     the payoff

    Returns:
        (symbols, rpe)
        
        symbols is a list of sympy symbols representing the components of the 
        Value RPE (i.e. Vdot_i)

        rpe is a list of sympy expressions representing the components of the
        Value RPE
    '''
    vdot_x = sympy.symarray(game.name + '_vdot', game.cardinality())
    rpes = []

    # [Isaacs, 4.6.2], 
    for k in range(game.cardinality()):
        eq = []
        for v, ke in zip(vdot_x, game.kinematic_eqns): 
            # V_i * f_ik
            eq.append(v * sympy.diff(ke, game.state[k]))

        rpes.append(sum(eq) + diff(payoff.G, game.state[k]))

    return vdot_x, rpes
# end _compute_value_rpe


def _compute_state_rpe(game):
    '''Compute the RPE of the state variables (xdot_k)

    The RPEs needed to solve the differential game consist of two sets:
        (1) the RPE of each component of the n-dimensional value function 
            (where n = game.cardinality())
        (2) the RPE of each componenet of the n-dimensional game space 
            (where n = game.cardinality())

    Note that (2) is simply the negative of the kinematic equations.

    This function computes (2) [Isaacs, Eqn 4.6.1]

    Arguments:
        game:       the game to process

    Returns:
        rpe
        
        rpe is a list of sympy expressions representing the components of the
        state RPE
    '''
    return [ - ke for ke in game.kinematic_equations ]
# end _compute_state_rpe


def solve(game, payoff, terminal):
    '''Solve the differential game.
   
    Solving the differential games consists of the following steps:
        (1) Computing the Retrograde Path Equations (RPEs)
        (2) Determining initial conditions for the RPEs based on the terminal
            surface
        (3) Solving the set of differential equations

    See [Isaacs, 4] for details, and [Jensen, 3.1] for an example.
    '''
    # (1) Compute RPEs
    vdot_k, value_rpes = _compute_value_rpe(game, payoff)
    state_rpes = _compute_state_rpe(game)

    # (2) Compute Initial Conditions
    

# end solve

