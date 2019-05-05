'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games

Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------

An example of using the solver to solve, analyze, and visualize a differential
game of simple linear motion.
'''

# Standard Imports
import argparse
import sys
import pickle as pkl

# External Imports
import numpy as np
import matplotlib.pyplot as plt
import rufus.game
import rufus.solver
import rufus.visualization

DT = 0.1

def progress(itr, max_itr):
    sys.stdout.write(f'\rIteration {itr} of {max_itr}')
# end progress

parser = argparse.ArgumentParser()
parser.add_argument(
        '--pursuer-speed',
        type=float,
        nargs=1,
        help='The puruser speed',
        default=3.0
)
parser.add_argument(
        '--evader-speed',
        type=float,
        nargs=1,
        help='The evader speed',
        default=1.0
)
parser.add_argument(
        '--iterations',
        type=int,
        help='The number of iterations to perform',
        default=100
)
parser.add_argument(
        '--capture-radius',
        type=int,
        nargs=1,
        help='The number of iterations to perform',
        default=5
)

args = parser.parse_args()

# capture set
def check_capture(v_p, v_e):
    return np.linalg.norm(v_e.loc - v_e.loc) < args.capture_radius
# end check_capture

# game space
region = rufus.game.BoxRegion(np.array([0.0, 0.0]), np.array([1000.0, 1000.0]))

# game kinematics
e_init  = rufus.game.Vertex(np.array([500.0, 500.0]), None, np.array([]))
evader  = rufus.game.LinearActor(DT, args.evader_speed)

p_init  = rufus.game.Vertex(np.array([100.0, 100.0]), None, np.array([]))
pursuer = rufus.game.LinearActor(DT, args.pursuer_speed)

# solve the game
solver = rufus.solver.Solver(
        DT,                 # time increment
        region,             # definition of game space
        pursuer,            # pursuer dynamics
        evader,             # evader dynamics
        check_capture,      # capture set definition
        gamma=500.0,       # scaling constant that defines how vertices are
                            # collapsed
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

rufus.visualization.plot_tree(soln.evader_tree(), c='b')
rufus.visualization.plot_tree(soln.pursuer_tree(), c='r')
plt.xlim([0.0, 1000.0])
plt.ylim([0.0, 1000.0])
plt.show()
