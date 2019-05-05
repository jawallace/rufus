'''
------------------------------------------------------------------------------
rufus - modeling and analysis of differential games

Jeffrey Wallace
EN.605.714, Spring 2019
------------------------------------------------------------------------------

This module contains utilities for analyzing games.
'''

# Standard Imports

# External Imports
import numpy as np

# Local Imports


class GameSolution:
    '''Represents a sampled solution to a pursuit-evasion game.'''

    def __init__(self, g_e, g_p):
        '''Constructor.

        Arguments:
            g_e: the evader trajectory graph
            g_p: the pursuer trajectory graph
        '''
        self._g_e = g_e
        self._g_p = g_p
    # end __init__


    def pursuer_tree(self):
        '''Get the pursuer's trajectory graph.'''
        return g_p
    # end pursuer_graph


    def evader_tree(self):
        '''Get the evader's trajectory graph.'''
        return g_p
    # end evader_graph


    def can_reach(self, target):
        '''Check if the evader can reach the given target region.

        Arguments:
            target: the target region to check

        Returns:
            True, if there is a path for the evader to reach the target
        '''
        return bool(self._reachable_nodes(target))
    # end can_reach

   
    def all_trajectories_to_target(self, target):
        '''Get all of the evader's trajectories that reach the target region.

        Arguments:
            target: the target region

        Returns:
            listof (path, trajectory)

            path is a list of Vertex that describe the trajectory segments,
            starting with the root node and ending with the node that passes
            through the trajectory

            trajectory is a np.ndarray that describes the trajectory of the
            evader over time

        Note:
            The returned path / trajectory may pass through the target area,
            not terminate within it. In other words, it is possible that

                path[-1].loc in target == False
                trajectory[-1] in target == False

            but it is guaranteed that, if this is the case, then

                any([pt in target for pt in path[-1].trajectory]) == True
                any([pt in target for pt in trajectory[-1]]) == True

        '''
        nodes = self._reachable_nodes(target)
        paths = list(map(self._collect_path, nodes))
        trajectories = list(map(self._collect_trajectory, paths))

        return list(zip(paths, trajectories))
    # end all_trajectories_to_target


    def min_trajectory_to_target(self, target):
        '''Return the fastest evader trajectory to the target.'''
        ts = self.all_trajectories_to_target(target)
        return (None, None) if not ts else min(ts, key=lambda p_t: p_t[1].shape[0])
    # end min_trajectory_to_target


    def max_time_trajectory(self):
        '''Return the evader trajectory of maximum time.
        
        Arguments:
            None

        Returns:
            (path, trajectory) of the trajectory of the largest duration

            path is a list of Vertex that describes the trajectory segments,
            starting with the root

            trajectory is a np.ndarray that describes the trajectory of the
            evader over time
        '''
        nodes = self._g_e.leaves()
        paths = list(map(self._collect_path, nodes))
        trajectories = list(map(self._collect_trajectory, paths))

        zipped = list(zip(paths, trajectories))
        return max(zipped, key=lambda p_t: p_t[1].shape[0])
    # end max_time_trajectory


    def _reachable_nodes(self, target):
        '''Get all nodes whose trajectories pass through the target.'''
        def _chk_pt(pt):
            return pt in target
        
        def _chk_node(n):
            if n.is_root():
                return _chk_pt(n.data.loc)
            else:
                return (
                    np.any(np.apply_along_axis(_chk_pt, 1, n.data.trajectory)) or
                    _chk_pt(n.data.loc)
                )

        nodes = list(self._g_e.filter_nodes(lambda n: _chk_node(n)))

        # filter out nodes who have an ancestor in nodes. if this happens, there is
        # already a node earlier in the trajectory that passes through the target

        # this is horribly inefficient :(
        # but, works for now
        to_remove = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue

                if self._g_e.is_ancestor(nodes[i].identifier, nodes[j].identifier):
                    # i is an ancestor of j, so we should remove j
                    to_remove.append(j)

        to_remove = sorted(set(to_remove), reverse=True)
        for idx in to_remove:
            del nodes[idx]
        return nodes
    # end _reachable_nodes


    def _collect_path(self, n):
        path = []
        cur = n
        while not cur.is_root():
            path.append(cur.data)
            cur = self._g_e.parent(cur.identifier)

        # add root
        path.append(cur.data)
        return list(reversed(path))
    # end _collect_path


    def _collect_trajectory(self, path):
        # condition in list comprehension is to ignore empty root trajectory
        return np.vstack([v.trajectory for v in path if v.trajectory.size] + [path[-1].loc])
    # end _collect_trajectory


# end GameSolution

