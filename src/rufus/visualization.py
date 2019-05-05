'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games

Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------

Visualization tools for rufus.
'''

# Standard Imports

# External Imports
import numpy as np
import matplotlib.pyplot as plt

# Local Imports
from rufus.analysis import GameSolution


def plot_trajectory(path, trajectory, **kwargs):
    '''Plot a single trajectory.'''
    if trajectory.shape[1] == 2:
        _plot_trajectory_2d(path, trajectory, **kwargs)
    elif trajectory.shape[1] == 3:
        _plot_trajectory_3d(path, trajectory, **kwargs)
    else:
        raise NotImplementedError('> 3D is not supported')
# end plot_trajectory


def _plot_trajectory_2d(path, trajectory, **kwargs):
    '''Plot a single 2d trajectory'''
    vertices = np.vstack([p.loc for p in path])
    plt.scatter(vertices[:, 0], vertices[:, 1], **kwargs)
    plt.plot(trajectory[:, 0], trajectory[:, 1], **kwargs)
# end _plot_trajectory_2d


def _plot_trajectory_3d(path, trajectory, **kwargs):
    '''Plot a single 3d trajectory'''
    vertices = np.vstack([p.loc for p in path])
    plt.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], **kwargs)
    plt.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], **kwargs)
# end _plot_trajectory_3d


def plot_tree(tree, **kwargs):
    vertices = []
    is_3d = None

    for node in tree.all_nodes_itr():
        if is_3d is None:
            if node.data.loc.shape[0] == 2:
                is_3d = False
            elif node.data.loc.shape[0] == 3:
                is_3d = True
            else:
                raise NotImplementedError(">3D not supported")

        trajectory = node.data.trajectory
        if trajectory.size:
            if is_3d:
                plt.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], **kwargs)
            else:
                plt.plot(trajectory[:, 0], trajectory[:, 1], **kwargs)

        vertices.append(node.data.loc)

    vertices = np.vstack(vertices)

    if is_3d:
        plt.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], **kwargs)
    else:
        plt.scatter(vertices[:, 0], vertices[:, 1], **kwargs)
# end plot_tree

