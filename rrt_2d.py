import argparse
from algorithm.rrt_algorithm import RRT
from algorithm.search_space import space
import numpy as np

# Set the random seed for reproducibility of results
np.random.seed(14)

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run the RRT algorithm with optional live plotting and ending plot."
)
parser.add_argument(
    "--live",
    type=lambda x: x.lower() == "true",
    default=True,
    help="Enable live plotting (True or False). Default is True.",
)
parser.add_argument(
    "--plot_result",
    type=lambda x: x.lower() == "true",
    default=True,
    help="Enable plotting result (True or False). Default is True.",
)
args = parser.parse_args()

# Define the dimensions of the 2D workspace (width, height)
dimensions = np.array([100, 100])

# Specify the starting coordinates within the workspace
start = np.array([1, 1])

# Specify the goal coordinates within the workspace
goal = np.array([99, 99])

# Define the acceptable radius around the goal to consider as reached
goal_radius = 3

# Set the maximum distance the tree can extend in one iteration
step_size = 7.395913334762972

# Define the maximum turning angle in degrees
theta = 238.9678587516086

# Set the chance to turn a sample into the theta range from goal to parent node
turn_percent = 64.50326182105157

# Set the percentage bias towards sampling the goal directly
bias_percent = 8.718183537094998

# Specify the total number of samples to be generated during RRT execution
n_samples = 2000

# Define the number of rectangular obstacles to be placed in the workspace
n_rectangles = 75

# Specify the range of sizes for the rectangular obstacles:
# First row: (min_width, max_width), Second row: (min_height, max_height)
rect_sizes = np.array([[5, 15], [5, 15]])

# Initialize the search space with the defined parameters
rrt_space = space(
    dimensions=dimensions,
    start=start,
    goal=goal,
    goal_radius=goal_radius,
    n_samples=n_samples,
    n_rectangles=n_rectangles,
    rect_sizes=rect_sizes,
)

# Instantiate the RRT algorithm with the configured search space
# The 'live' parameter enables real-time visualization during execution
# The 'plot_result' parameter enables plotting after execution of the algorithm
rrt_algorithm = RRT(
    rrt_space,
    step_size=step_size,
    theta=theta,
    turn_chance=turn_percent / 100.0,  # Convert percentage to a decimal
    bias_chance=bias_percent / 100.0,  # Convert percentage to a decimal
    live=args.live,
    plot_result=args.plot_result,
)

# Execute the RRT algorithm
# Returns:
# - found_path: if the algorithm found a path in the space constraints
# - num_samples: number of samples it took to find a path to the goal
found_path, num_samples, n_tries_to_place_node, path_distance = rrt_algorithm.execute()

print(f"Found Path: {found_path}")
print(f"Path Distance: {path_distance}")
print(f"Number of samples: {num_samples}")
