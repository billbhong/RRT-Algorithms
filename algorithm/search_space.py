import numpy as np
from utilities.geometry import dist_between_points


class space(object):
    """
    Represents a 2D search space for path planning algorithms, including
    workspace dimensions, start and goal positions, obstacles, and relevant
    parameters for algorithms like RRT.
    """

    def __init__(
        self,
        dimensions: np.ndarray,
        start: np.ndarray,
        goal: np.ndarray,
        goal_radius: float = 1.0,
        n_samples: int = 1000,
        n_rectangles: int = 10,
        rect_sizes: np.ndarray = np.zeros((2, 2)),
    ):
        """
        Initializes the search space with specified parameters and generates obstacles.

        Parameters:
            dimensions (np.ndarray): The width and height of the workspace.
            start (np.ndarray): The starting position coordinates.
            goal (np.ndarray): The goal position coordinates.
            goal_radius (float): The radius within which the goal is considered reached.
            step_size (float): The maximum extension length for path planning steps.
            theta (float): The maximum turning angle in degrees.
            turn_chance (float): The chance to turn a sample into the theta range from goal to parent node.
            bias_chance (float): The probability of sampling the goal directly.
            n_samples (int): The number of samples to generate for path planning.
            n_rectangles (int): The number of rectangular obstacles to generate.
            rect_sizes (np.ndarray): The size range for the rectangular obstacles.
        """
        self.dimensions = dimensions
        self.start = start
        self.goal = goal
        self.goal_radius = goal_radius
        self.n_samples = n_samples
        self.n_rectangles = n_rectangles
        self.rect_sizes = rect_sizes
        self.generate_obstacles()

    def generate_obstacles(self):
        """
        Generates random rectangular obstacles and adds border obstacles to the search space.
        """
        self.rectangles = self._generate_rectangles()
        self.rectangles = np.append(self.rectangles, self._generate_border(), axis=1)

    def _generate_border(self):
        """
        Generates border obstacles around the workspace to prevent paths from exiting the boundaries.

        Returns:
            A numpy array in the shape (4,4) representing the four rectangle obstacle borders.
        """

        x_dim_max = self.dimensions[0]
        y_dim_max = self.dimensions[1]

        thickness = 0.1

        x_min = np.array([0, x_dim_max, 0, -thickness])
        x_max = np.array([x_dim_max, x_dim_max + thickness, x_dim_max, 0])

        y_min = np.array([-thickness, 0, y_dim_max, 0])
        y_max = np.array([0, y_dim_max, y_dim_max + thickness, y_dim_max])

        return np.array([x_min, y_min, x_max, y_max])

    def _generate_rectangles(self):
        # Initial generation
        widths, heights = self._generate_rect_sizes(self.n_rectangles, self.rect_sizes)
        x_min, y_min = self._generate_rect_starting_pose(widths, heights)
        x_max = x_min + widths
        y_max = y_min + heights

        # Keep regenerating bad rectangles until not covering start or goal
        while True:
            # Check the rectangles that cover the start or goal
            covers_start = (
                (self.start[0] >= x_min)
                & (self.start[0] <= x_max)
                & (self.start[1] >= y_min)
                & (self.start[1] <= y_max)
            )
            covers_goal = (
                (self.goal[0] >= x_min)
                & (self.goal[0] <= x_max)
                & (self.goal[1] >= y_min)
                & (self.goal[1] <= y_max)
            )
            invalid = covers_start | covers_goal

            if not invalid.any():
                break

            # resample only the bad ones
            bad = np.nonzero(invalid)[0]
            nb = bad.size

            # widths/heights -> new x_min,y_min -> new x_max,y_max
            w_new, h_new = self._generate_rect_sizes(nb, self.rect_sizes)
            x_min_new, y_min_new = self._generate_rect_starting_pose(w_new, h_new)

            widths[bad] = w_new
            heights[bad] = h_new
            x_min[bad] = x_min_new
            y_min[bad] = y_min_new
            x_max[bad] = x_min_new + w_new
            y_max[bad] = y_min_new + h_new

        return np.array([x_min, y_min, x_max, y_max])

    def _generate_rect_sizes(self, n: int, rect_sizes):
        """
        Sample n widths/heights from rect_sizes.
        Returns two arrays of shape (n,): widths, heights.
        """
        (w_min, w_max), (h_min, h_max) = rect_sizes
        widths = np.random.rand(n) * (w_max - w_min) + w_min
        heights = np.random.rand(n) * (h_max - h_min) + h_min
        return widths, heights

    def _generate_rect_starting_pose(self, widths, heights):
        """
        Given arrays widths and heights of length n,
        sample bottom-left corners so each rect fits in [0..bounds].
        Returns two arrays (x_min, y_min) of shape (n,).
        """
        x_min = np.random.rand(widths.size) * (self.dimensions[0] - widths)
        y_min = np.random.rand(heights.size) * (self.dimensions[1] - heights)
        return x_min, y_min

    def collision_free_path(self, start: np.ndarray, end: np.ndarray) -> bool:
        """
        Determine whether a straight-line path between two points is free of collisions
        with any axis-aligned rectangular obstacles in the environment.

        Parameters
        ----------
        start : np.ndarray
            A 1D array of shape (2,) representing the starting coordinates [x, y].
        end : np.ndarray
            A 1D array of shape (2,) representing the ending coordinates [x, y].

        Returns
        -------
        bool
            True if the path does not intersect any obstacle; False otherwise.
        """
        x_min, y_min, x_max, y_max = self.rectangles

        # Compute intersection points between the path and each side of all rectangles.
        # Each rectangle side is represented by its endpoints.
        intersection_points_line1 = self._calculate_intersection_points(
            start,
            end,
            x_min,
            y_min,
            x_max,
            y_min,  # Bottom side
        )
        intersection_points_line2 = self._calculate_intersection_points(
            start,
            end,
            x_max,
            y_min,
            x_max,
            y_max,  # Right side
        )
        intersection_points_line3 = self._calculate_intersection_points(
            start,
            end,
            x_max,
            y_max,
            x_min,
            y_max,  # Top side
        )
        intersection_points_line4 = self._calculate_intersection_points(
            start,
            end,
            x_min,
            y_max,
            x_min,
            y_min,  # Left side
        )

        # Define a sentinel value for non-intersecting lines (parallel or coincident).
        max_val = np.finfo(np.float64).max

        # Check if any intersection computation resulted in the sentinel value,
        # indicating parallel or coincident lines, which are treated as collisions.
        if (
            np.any(intersection_points_line1 == max_val)
            or np.any(intersection_points_line2 == max_val)
            or np.any(intersection_points_line3 == max_val)
            or np.any(intersection_points_line4 == max_val)
        ):
            return False

        # Extract x and y coordinates of intersection points for each rectangle side.
        px1, py1 = intersection_points_line1
        px2, py2 = intersection_points_line2
        px3, py3 = intersection_points_line3
        px4, py4 = intersection_points_line4

        # Determine if any intersection point lies within the bounds of the corresponding
        # rectangle side and between the start and end points of the path.
        mask1 = (
            (x_min <= px1)
            & (px1 <= x_max)
            & (np.minimum(start[1], end[1]) <= py1)
            & (py1 <= np.maximum(start[1], end[1]))
        )
        mask2 = (
            (y_min <= py2)
            & (py2 <= y_max)
            & (np.minimum(start[0], end[0]) <= px2)
            & (px2 <= np.maximum(start[0], end[0]))
        )
        mask3 = (
            (x_min <= px3)
            & (px3 <= x_max)
            & (np.minimum(start[1], end[1]) <= py3)
            & (py3 <= np.maximum(start[1], end[1]))
        )
        mask4 = (
            (y_min <= py4)
            & (py4 <= y_max)
            & (np.minimum(start[0], end[0]) <= px4)
            & (px4 <= np.maximum(start[0], end[0]))
        )

        # If any intersection point lies within a rectangle side and between the start
        # and end points of the path, the path is considered to be in collision.
        if np.any(mask1 | mask2 | mask3 | mask4):
            return False

        # If no collisions are detected, the path is considered collision-free.
        return True

    def _calculate_intersection_points(
        self,
        start: np.ndarray,
        end: np.ndarray,
        x_3: np.ndarray,
        y_3: np.ndarray,
        x_4: np.ndarray,
        y_4: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the intersection point between a line segment defined by `start` and `end`
        and another line segment defined by endpoints (x_3, y_3) and (x_4, y_4), using
        the standard line-line intersection formula.

        If the lines are parallel or coincident (i.e., no unique intersection exists),
        a large sentinel value is returned to indicate an invalid intersection.
        
        Calculation from: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

        Parameters:
            start (np.ndarray): 1D array of shape (2,) representing the first endpoint of the first line.
            end (np.ndarray): 1D array of shape (2,) representing the second endpoint of the first line.
            x_3, y_3 (np.ndarray): Coordinates of the first endpoint of the second line.
            x_4, y_4 (np.ndarray): Coordinates of the second endpoint of the second line.

        Returns:
            np.ndarray: A 1D array of shape (2,) representing the (x, y) coordinates of the
                        intersection point, or [max_float] if no intersection exists.
        """

        # Unpack coordinates for readability
        x_1, y_1 = start[0], start[1]
        x_2, y_2 = end[0], end[1]

        # Compute the denominator of the intersection formula
        P_xy_denom = (x_1 - x_2) * (y_3 - y_4) - (y_1 - y_2) * (x_3 - x_4)

        # If denominator is zero, the lines are parallel or coincident
        if (P_xy_denom == 0).any():
            return np.array([np.finfo(np.float64).max])

        # Compute numerators for x and y coordinates of the intersection point
        P_x_numer = (x_1 * y_2 - y_1 * x_2) * (x_3 - x_4) - (x_1 - x_2) * (
            x_3 * y_4 - y_3 * x_4
        )
        P_y_numer = (x_1 * y_2 - y_1 * x_2) * (y_3 - y_4) - (y_1 - y_2) * (
            x_3 * y_4 - y_3 * x_4
        )

        # Return the intersection point
        return np.array([P_x_numer / P_xy_denom, P_y_numer / P_xy_denom])

    def close_to_goal(self, pose: np.ndarray):
        """
        Checks whether a given pose is within the goal radius.

        Parameters:
            pose (np.ndarray): The position to check.

        Returns:
            bool: True if the pose is within the goal radius; False otherwise.
        """
        return dist_between_points(pose, self.goal) <= self.goal_radius
