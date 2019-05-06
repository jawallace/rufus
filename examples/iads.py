'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games

Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------

An example of using the solver to solve, analyze, and visualize the
IADS differential game.

This version uses random initial positions and estimates the solution to the
game of kind (can the evader reach the target?) and degree (how close can the
evader get to the target)
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
        '--attacker-speed',
        type=float,
        help='The evader speed, relative to the pursuer. Must be in range (0, 1]',
        default=0.3
)
parser.add_argument(
        '--attacker-start-location',
        type=float,
        nargs=3,
        help='The attacker starting location',
        default=[50.0, 0.0, 30.0]
)
parser.add_argument(
        '--defender-start-location',
        type=float,
        nargs=2,
        help='The defender starting location',
        default=[0.0, 70.0, 0.0]
)
parser.add_argument(
        '--defender-start-direction',
        type=float,
        help=(
            'Starting azimuth of the defender, in degrees relative to the '
            'positive x-axis. Valid range is (-180, 180]. '
        ),
        default=-45.0
)
parser.add_argument(
        '--defender-max-bank',
        type=float,
        help='The defenders maximum bank angle in degrees.',
        default=45.0
)
parser.add_argument(
        '--defender-max-gamma',
        type=float,
        help='The defenders maximum gamma angle in degrees.',
        default=30.0
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
parser.add_argument(
        '--target-lower',
        type=float,
        nargs=3,
        help='The lower bound of the rectangular target region',
        default=[25.0, 80.0, 0.0]
)
parser.add_argument(
        '--target-upper',
        type=float,
        nargs=3,
        help='The upper bound of the rectangular target region',
        default=[75.0, 100.0, 50.0]
)
parser.add_argument(
        '--output',
        type=str,
        help='The file in which to save the game solution',
        default='iads.soln.pkl'
)

args = parser.parse_args()

# capture set
def check_capture(v_p, v_e):
    '''For the IADS game, the capture conditions are as follows:

        (1) the distance between the evader and pursuer < capture radius
        (2) the pursuer is 'pointing' at the evader

    Condition (2) prevents the unphysical scenario of the defender (which can
    only move forward) 'capturing' the attacker when the attacker is behind the
    defender. This is the analog of Isaacs' 'Usable Part' and is the same concept
    as the Homicidal Chauffeur's capture set.
    '''
    diff = v_e.loc - v_p.loc

    azimuth     = np.arctan(diff[1] / diff[0])
    elevation   = np.arctan(diff[2] / np.sqrt(np.sum(diff[:2]**2)))

    dist = np.sqrt(np.sum(diff**2))

    # we do not need to check the elevation angle. An invariant of the Dubin's
    # Airplane implementation that we use is that the elevation angle is always 0
    # at a vertex - so the target will always be in the usable elevation range,
    # provided it is in the usable azimuth range
    return (dist < args.capture_radius) and (np.abs(azimuth - v_p.state) < np.pi)
# end check_capture

# game space
region = rufus.game.BoxRegion(
        np.array([0.0, 0.0, 0.0]),
        np.array([args.dimension, args.dimension, args.dimension])
)

# target area
target = rufus.game.BoxRegion(
        np.array(args.target_lower),
        np.array(args.target_upper)
)

# The evader is highly maneuvarable, but slower than the pursuer
e_init = rufus.game.Vertex(
        np.array(args.attacker_start_location),         # initial location
        None,                                           # initial state
        np.array([])                                    # empty trajectory for root node
)
evader  = rufus.actors.LinearActor(args.dt, args.attacker_speed)

# The pursuer is faster, but less maneuverable
p_init = rufus.game.Vertex(
        np.array(args.defender_start_location),         # initial location
        np.deg2rad(args.defender_start_direction),      # initial state
        np.array([])                                    # empty trajectory for root node
)
pursuer = rufus.actors.DubinsAirplane(
        args.dt,
        np.deg2rad(args.defender_max_bank),
        np.deg2rad(args.defender_max_gamma),
        1.0
)

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

with open(args.output, 'wb') as fid:
    pkl.dump(soln, fid)


#################################################################################
# Game of Kind - Can the attacker reach the target?
value = soln.can_reach(target)
print()
print(f'Game of Kind Solution - Can the attacker reach the target?: {value}')

if value:
    # plot all reachable trajectories, including the shortest-path
    results = soln.all_trajectories_to_target(target)
    mpath, mtrajectory = soln.min_trajectory_to_target(target)

    ax = plt.axes(projection='3d')
    for path, trajectory in results:
        rufus.visualization.plot_trajectory(path, trajectory, ax=ax, c='b', alpha=0.5)

    rufus.visualization.plot_trajectory(mpath, mtrajectory, ax=ax, c='g', linewidth=4.0, vertexweight=4)

    rufus.visualization.plot_region(target, linewidths=1, edgecolors='r', alpha=0.25, facecolors='r', ax=ax)

    rufus.visualization.plot_vector(
            p_init.loc,
            np.array([np.cos(p_init.state), np.sin(p_init.state), 0]),
            ax=ax
    )
    plt.title(f'Solution to the Game of Kind. ({args.dt * mtrajectory.shape[0]:0.2f} s)')
    plt.show()

rufus.visualization.plot_tree(soln.pursuer_tree(), samples=20, c='r', alpha=0.5)
plt.show()
