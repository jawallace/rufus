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
from mpl_toolkits import mplot3d

# Local Imports
from rufus.analysis import GameSolution
from rufus.game import BoxRegion


def plot_vector(loc, direction, **kwargs):
    if loc.shape[0] == 2:
        _plot_vector_2d(loc, direction, **kwargs)
    else:
        _plot_vector_3d(loc, direction, **kwargs)
# end plot_vector


def _plot_vector_2d(loc, direction, **kwargs):
    plt.arrow(loc[0], loc[1], direction[0], direction[1], **kwargs)
# end _plot_vector_2d


def _plot_vector_3d(loc, direction, ax=None, **kwargs):
    if ax is None:
        axes = plt.axes(projection='3d')
    else:
        axes = ax

    axes.quiver(
            loc[0], loc[1], loc[2],
            direction[0], direction[1], direction[2],
            **kwargs
    )

    return axes
# end _plot_vector_3d


def plot_region(region, **kwargs):
    assert isinstance(region, BoxRegion)

    if region.ndim == 2:
        _plot_region_2d(region, **kwargs)
    elif region.ndim == 3:
        _plot_region_3d(region, **kwargs)
    else:
        raise NotImplementedError('>3D not supported')
# end plot_region


def _plot_region_2d(region, **kwargs):
    pass
# end _plot_region_2d


def _plot_region_3d(region, ax=None, **kwargs):
    if ax is None:
        axes = plt.axes(projection='3d')
    else:
        axes = ax

    x0, y0, z0 = region.lower
    x1, y1, z1 = region.upper

    pts = np.array([
        [x0, y0, z0],
        [x0, y0, z1],
        [x0, y1, z0],
        [x0, y1, z1],
        [x1, y0, z0],
        [x1, y0, z1],
        [x1, y1, z0],
        [x1, y1, z1],
    ])

    sides = np.array([
        # y-axis fixed
        [pts[0], pts[1], pts[5], pts[4]],
        [pts[2], pts[3], pts[7], pts[6]],

        # x-axis fixed
        [pts[0], pts[1], pts[3], pts[2]],
        [pts[4], pts[5], pts[7], pts[6]],

        # z-axis fixed
        [pts[0], pts[2], pts[6], pts[4]],
        [pts[1], pts[3], pts[7], pts[5]]
    ])

    axes.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2], **kwargs)
    fc = None
    if 'facecolors' in kwargs:
        fc = kwargs['facecolors']
        del kwargs['facecolors']

    coll = mplot3d.art3d.Poly3DCollection(sides, **kwargs)
    if fc is not None:
        coll.set_facecolor(fc)

    axes.add_collection3d(coll)

    return axes
# end _plot_region_3d


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


def _plot_trajectory_3d(path, trajectory, ax=None, vertexweight=1, **kwargs):
    '''Plot a single 3d trajectory'''
    if ax is None:
        axes = plt.axes(projection='3d')
    else:
        axes = ax

    vertices = np.vstack([p.loc for p in path])
    axes.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            s=vertexweight,
            **kwargs
    )
    axes.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], **kwargs)

    return ax
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
    ax = None

    for node in nodes:
        if is_3d is None:
            if node.data.loc.shape[0] == 2:
                is_3d = False
            elif node.data.loc.shape[0] == 3:
                is_3d = True
                ax = kwargs.get('ax', plt.axes(projection='3d'))
            else:
                raise NotImplementedError(">3D not supported")

        trajectory = node.data.trajectory
        if trajectory.size:
            if is_3d:
                ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], **kwargs)
            else:
                plt.plot(trajectory[:, 0], trajectory[:, 1], **kwargs)

        vertices.append(node.data.loc)

    vertices = np.vstack(vertices)

    if is_3d:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], **kwargs)
        return ax
    else:
        plt.scatter(vertices[:, 0], vertices[:, 1], **kwargs)
# end plot_nodes

