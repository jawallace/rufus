'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games

Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------

An example of using the solver to solve, analyze, and visualize the
Homicidal Chauffeur differential game.

This version uses random initial positions and estimates the solution to the
game of kind (can the pursuer capture the evader?) and degree
(how long can the evader avoid capture?).
'''

# Standard Imports
import argparse
import sys
import pickle as pkl

# External Imports
import matplotlib.pyplot as plt
import numpy as np
import rufus.actors
import rufus.game
import rufus.solver
import rufus.visualization

def progress(itr, max_itr):
    sys.stdout.write(f'\rIteration {itr} of {max_itr}')
# end progress

parser = argparse.ArgumentParser()
parser.add_argument(
        '--dt',
        type=float,
        help='The sampling period, in seconds',
        default=0.1
)
parser.add_argument(
        '--dimension',
        type=float,
        help='The dimension of the square, 2D gamespace',
        default=100.0
)
parser.add_argument(
        '--evader-speed',
        type=float,
        help='The evader speed, relative to the pursuer. Must be in range (0, 1]',
        default=0.3
)
parser.add_argument(
        '--pursuer-start-location',
        type=float,
        nargs=2,
        help='The pursuers starting location',
        default=[0.0, 0.0]
)
parser.add_argument(
        '--pursuer-start-direction',
        type=float,
        help=(
            'Starting orientation of the pursuer, in degrees relative to the '
            'positive x-axis. Valid range is (-180, 180]. '
        ),
        default=45.0
)
parser.add_argument(
        '--pursuer-turning-radius',
        type=float,
        help='The pursuer turning radius. Must be in range (0, 20.0]',
        default=10.0
)
parser.add_argument(
        '--iterations',
        type=int,
        help='The number of iterations to perform',
        default=2500
)
parser.add_argument(
        '--capture-radius',
        type=int,
        help='The radius of capture',
        default=5
)

args = parser.parse_args()

# capture set
def check_capture(v_p, v_e):
    '''For the Homicidal Chauffeur game, the capture conditions are as follows:

        (1) the distance between the evader and pursuer < capture radius
        (2) the pursuer is 'pointing' at the evader

    Condition (2) prevents the unphysical scenario of a Dubin's car (which can
    only move forward) hitting the evader with the back of the car. This is the
    analog of Isaacs' 'Usable Part'.
    '''
    diff = v_e.loc - v_p.loc
    dist = np.sqrt(diff[0]**2 + diff[1]**2)

    theta = np.arccos(diff[0] / dist)
    return (dist < args.capture_radius) and (np.abs(theta - v_p.state) < np.pi)
# end check_capture

# game space
region = rufus.game.BoxRegion(np.array([0.0, 0.0]), np.array([args.dimension, args.dimension]))

# The evader is highly maneuvarable, but slower than the pursuer
e_init = rufus.game.Vertex(
        np.array([args.dimension, args.dimension]) / 2, # initial location
        None,                                           # initial state
        np.array([])                                    # empty trajectory for root node
)
evader  = rufus.actors.LinearActor(args.dt, args.evader_speed)

# The pursuer is faster, but less maneuverable
p_init = rufus.game.Vertex(
        np.array(args.pursuer_start_location),          # initial location
        np.deg2rad(args.pursuer_start_direction),       # initial state
        np.array([])                                    # empty trajectory for root node
)
pursuer = rufus.actors.DubinsCar(args.dt, args.pursuer_turning_radius)

# solve the game
solver = rufus.solver.Solver(
        args.dt,                # time increment
        region,                 # definition of game space
        pursuer,                # pursuer dynamics
        evader,                 # evader dynamics
        check_capture,          # capture set definition
        gamma=args.dimension    # scaling constant that defines how vertices are
                                # collapsed. Should be ~ game space dimension
)

print(f'Starting solver ({args.iterations})')
soln = solver.solve(
        p_init,             # initial pursuer location
        e_init,             # initial evader location
        args.iterations,    # number of iterations
        progress=progress   # progress callback
)
print('\nDone')

with open('foo.pkl', 'wb') as fid:
    pkl.dump(soln, fid)

#################################################################################
# Game of Degree - How long can the evader prevent capture?
path, trajectory = soln.max_time_trajectory()
rufus.visualization.plot_tree(soln.evader_tree(), c='b', alpha=0.5)
rufus.visualization.plot_trajectory(path, trajectory, c='g', vertexweight=4, linewidth=4.0)
plt.arrow(
    p_init.loc[0],
    p_init.loc[1],
    args.dimension / 100 * np.cos(p_init.state),
    args.dimension / 100 * np.sin(p_init.state),
    width=args.dimension / 250,
    color='r',
    fill=True,
    head_starts_at_zero=True
)
plt.title(f'Game of Degree - Max Length Path ({args.dt * trajectory.shape[0]:0.2f} s)')
plt.xlim([0.0, args.dimension])
plt.ylim([0.0, args.dimension])
plt.show()

rufus.visualization.plot_tree(soln.pursuer_tree(), samples=20, c='r', alpha=0.5)
plt.show()
