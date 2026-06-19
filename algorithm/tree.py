from typing import Optional
import numpy as np
from treelib.tree import Tree
from algorithm.search_space import space
from utilities.geometry import steer, original_steer, dist_between_points


class rrt_tree(object):
    """
    Represents a Rapidly-Exploring Random Tree (RRT) for path planning.

    This class manages the construction and manipulation of an RRT within a defined search space,
    facilitating the addition of nodes and retrieval of paths.

    Attributes:
        tree (Tree): The underlying tree structure from the treelib library.
        node_count (int): Counter for the number of nodes added to the tree.
        space (space): The search space containing environment dimensions, obstacles, start, and goal positions.
    """

    def __init__(self, space: space, step_size: float, theta: float, turn_chance: float):
        """
        Initialize the RRT with the given search space.

        Args:
            space (space): The search space defining the environment for path planning.
        """
        self.space = space
        self.step_size = step_size
        self.theta = theta
        self.turn_chance = turn_chance
        self.node_count = 0
        self.tree = Tree()
        self.add_node(space.start)

    def add_node(self, pose: np.ndarray):
        """
        Attempt to add a new node to the RRT at the specified position.

        The method finds the nearest existing node in the tree, steers towards the new pose,
        and adds the new node if the path is collision-free.

        Args:
            pose (np.ndarray): The target position to add to the tree.

        Returns:
            Tuple[Optional[np.ndarray], Optional[str]]: A tuple containing the steered position and the node identifier.
            Returns (None, None) if the node could not be added due to collision or other constraints.
        """
        self.node_count += 1
        nid = f"Node {self.node_count}"
        steered_pose = pose

        if self.tree.size() == 0:
            self.tree.create_node(nid, nid, data=ArrayHolder(pose))
            return pose, nid
        else:
            parent_node = None
            min_distance = np.finfo(np.float64).max

            for node in self.tree.all_nodes_itr():
                # skip any tree‐nodes that somehow have no data
                if node.data is None:
                    continue

                d = dist_between_points(node.data.array, pose)

                if d < min_distance:
                    min_distance = d
                    parent_node = node
                    steered_pose = steer(
                        parent_node.data.array,
                        pose,
                        self.step_size,
                        self.space.goal,
                        self.theta,
                        self.turn_chance
                    )

                    # steered_pose = original_steer(
                    #     parent_node.data.array, pose, self.space.step_size
                    # )

            # ensure we actually found a parent before dereferencing
            if parent_node is not None and self.space.collision_free_path(
                steered_pose, parent_node.data.array
            ):
                self.tree.create_node(
                    nid,
                    nid,
                    parent=parent_node,
                    data=ArrayHolder(steered_pose),
                )

                return steered_pose, nid

            self.node_count -= 1
            return None, None
    
    def distance_to_node_from_root(self, path_nids) -> float:
        """
        Calculate the total distance from the root node to the specified node.

        Args:
            node_nid (str): The identifier of the target node.

        Returns:
            float: The total distance from the root node to the target node.
        """

        total_distance = 0.0
        for i in range(len(path_nids) - 1):
            current_node_id = path_nids[i]
            next_node_id = path_nids[i+1]
            
            current_pose = self.tree[current_node_id].data.array
            next_pose = self.tree[next_node_id].data.array
            
            total_distance += dist_between_points(current_pose, next_pose)
            
        return total_distance # type: ignore

    def path_to_node(self, nid: str) -> list[str]:
        """
        Retrieve the sequence of node identifiers from the root to the specified node.

        Args:
            nid (str): The identifier of the target node.

        Returns:
            List[str]: A list of node identifiers representing the path from the root to the target node.
        """
        seq = list(self.tree.rsearch(nid=nid))
        return list(reversed(seq))

    def show(self, data_property: Optional[str] = None):
        """
        Display the structure of the tree.

        Args:
            data_property (Optional[str]): An optional property name to display for each node.

        Returns:
            None
        """
        return self.tree.show(data_property=data_property)

    def size(self):
        """
        Get the number of nodes in the tree.

        Returns:
            int: The total number of nodes in the tree.
        """
        return self.tree.size()
    
    def reset(self):
        self.tree = Tree()
        self.add_node(self.space.start)


class ArrayHolder:
    """
    A simple wrapper class for storing numpy arrays for use in nodes in the RRT tree.

    Attributes:
        array (np.ndarray): The numpy array being wrapped.
    """

    def __init__(self, arr):
        """
        Initialize the ArrayHolder with the given array.

        Args:
            arr (np.ndarray): The array to be stored.
        """
        self.array = arr

    def __repr__(self):
        """
        Return a string representation of the stored array.

        Returns:
            str: A string representation of the array.
        """
        return np.array2string(self.array, separator=", ")
