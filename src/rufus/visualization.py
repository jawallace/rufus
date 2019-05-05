'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games

Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------

Visualization tools for rufus.
'''

# Standard Imports
import random

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


def _plot_trajectory_2d(path, trajectory, vertexweight=0.25, **kwargs):
    '''Plot a single 2d trajectory'''
    vertices = np.vstack([p.loc for p in path])
    plt.scatter(vertices[:, 0], vertices[:, 1], vertexweight, **kwargs)
    plt.plot(trajectory[:, 0], trajectory[:, 1], **kwargs)
# end _plot_trajectory_2d


def _plot_trajectory_3d(path, trajectory, **kwargs):
    '''Plot a single 3d trajectory'''
    vertices = np.vstack([p.loc for p in path])
    plt.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], **kwargs)
    plt.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], **kwargs)
# end _plot_trajectory_3d


def plot_tree(tree, samples=None, **kwargs):
    '''Plot a tree.

    If samples is None, all branches will be plotted. Otherwise, ``samples``
    branches of the tree will be randomly selected and plotted.

    Arguments:
        tree:       the tree to plot
        samples:    the number of branches to plot
        **kwargs:   keyword arguments to be passed to matplotlib

    Returns:
        None

    Postcondition:
        the active matplotlib plot will be populated with the tree
        visualization.
    '''
    if samples is None:
        _plot_nodes(tree.all_nodes_itr(), **kwargs)
        return

    leaves = list(tree.leaves())
    random.shuffle(list(tree.leaves()))

    selected = leaves[:samples]
    nodes = []
    for l in selected:
        cur = l
        while not cur.is_root():
            nodes.append(cur)
            cur = tree.parent(cur.identifier)

    nodes.append(tree[tree.root])
    nodes = set(nodes)

    _plot_nodes(nodes, **kwargs)
# end plot_tree


def _plot_nodes(nodes, **kwargs):
    vertices = []
    is_3d = None

    for node in nodes:
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
# end plot_nodes

