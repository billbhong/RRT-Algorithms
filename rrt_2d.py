import argparse
import json
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
parser.add_argument(
    "--n_trials",
    type=int,
    default=1,
    help="Number of trails to run. Default is 1",
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
n_rectangles = 65

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

# Accumulators for aggregating results across all trials
stats = {}            # Per-trial stats, keyed by trial number
num_success = 0       # Count of trials that successfully found a path
avg_distance = 0.0    # Running sum of path distances (averaged after the loop)
avg_num_samples = 0.0 # Running sum of sample counts (averaged after the loop)

# Run the RRT algorithm once per trial, regenerating obstacles between trials
for i in range(args.n_trials):
    print(f"Starting next trail: {i+1}\n")

    # Execute the RRT algorithm
    # Returns:
    # - found_path: if the algorithm found a path in the space constraints
    # - num_samples: number of samples it took to find a path to the goal
    # - n_tries_to_place_node: number of attempts needed to place valid nodes
    # - path_distance: total length of the path found (0 if no path)
    found_path, num_samples, n_tries_to_place_node, path_distance = rrt_algorithm.execute()

    print(f"Found Path: {found_path}")
    print(f"Path Distance: {path_distance}")
    print(f"Number of samples: {num_samples}\n\n")

    # Tally a success only when a valid path to the goal was found
    if found_path:
        num_success += 1

    # Accumulate this trial's metrics into the running totals
    avg_distance += path_distance
    avg_num_samples += num_samples

    # Record this trial's results so they can be inspected/exported later
    stats[i + 1] = {
        "found_path": found_path,
        "num_samples": num_samples,
        "n_tries_to_place_node": n_tries_to_place_node,
        "path_distance": path_distance,
    }

    # Generate a fresh set of obstacles so the next trial runs on a new map
    print("Generaing next trial obtacles\n")
    rrt_space.generate_obstacles()
    rrt_algorithm.reset()

# Convert the accumulated sums into per-trial averages
avg_distance /= args.n_trials
avg_num_samples /= args.n_trials

# print total summary, append final stats, and save stats to json
summary = {
    "n_trials": args.n_trials,
    "num_success": num_success,
    "success_rate": num_success / args.n_trials,
    "avg_distance": avg_distance,
    "avg_num_samples": avg_num_samples,
}

print("===== Summary =====")
print(f"Trials run:        {summary['n_trials']}")
print(f"Successful paths:  {summary['num_success']}")
print(f"Success rate:      {summary['success_rate'] * 100:.2f}%")
print(f"Average distance:  {summary['avg_distance']:.4f}")
print(f"Average samples:   {summary['avg_num_samples']:.2f}")

# Append the aggregate summary alongside the per-trial results
stats["summary"] = summary

# Save all collected stats to a JSON file for later analysis
with open("rrt_2d_stats65.json", "w") as f:
    json.dump(stats, f, indent=4)

print("\nStats saved to rrt_2d_stats.json")