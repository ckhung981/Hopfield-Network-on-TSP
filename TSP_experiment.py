import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from itertools import permutations
import os
import datetime

class HopfieldTSPSolver:
    """
    A class to solve the Traveling Salesperson Problem (TSP) using a Hopfield Network.
    Simulated Annealing can be optionally enabled for better optimization.
    """
    def __init__(self, num_cities, city_coords, params=None, use_simulated_annealing=True):
        """
        Initializes the TSP solver.

        Args:
            num_cities (int): The number of cities.
            city_coords (np.ndarray): (x, y) coordinates of the cities, shape (num_cities, 2).
            params (dict, optional): A dictionary containing parameters for the Hopfield Network
                                     and Simulated Annealing. If None, default parameters are used.
            use_simulated_annealing (bool): If True, apply simulated annealing (temperature cools).
                                             If False, temperature remains constant at T_initial.
        """
        self.n = num_cities
        self.city_coords = city_coords
        self.distances = self._calculate_distance_matrix(city_coords)

        # Default parameters for the Hopfield network and simulated annealing
        self.params = {
            'A': 500,
            'B': 500,
            'D': 10,
            'C': 500 / num_cities,  # Default C is often A/n
            'T_initial': 1.0,
            'T_final': 0.01,
            'alpha': 0.99995,
            'max_iterations': 150000,
            'random_factor': 50
        }
        # Update default parameters if custom parameters are provided
        if params:
            self.params.update(params)
        
        self.use_simulated_annealing = use_simulated_annealing

        # Initialize the neuron matrix V with small random values around 1/n
        # This will be re-initialized if solve() is called multiple times in a loop
        self.V = np.random.rand(self.n, self.n) * 0.2 + (1/self.n - 0.1)

        # Pre-compute the weight matrix W and bias vector b as they are fixed
        self.W = self._compute_weights()
        self.b = self._compute_bias()

        self.best_path_hopfield = None
        self.best_length_hopfield = float('inf')

    def _calculate_distance_matrix(self, coords):
        """Calculates the Euclidean distance matrix between cities."""
        num_cities = coords.shape[0]
        dist_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                dist = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist # Ensure symmetry
        return dist_matrix

    def _compute_weights(self):
        """
        第一個索引 x (city index): 代表目標神經元 (target neuron) 所對應的城市。
        第二個索引 i (position index): 代表目標神經元所對應的旅程位置。
        第三個索引 y (city index): 代表來源神經元 (source neuron) 所對應的城市。
        第四個索引 j (position index): 代表來源神經元所對應的旅程位置。
        Computes the four-dimensional weight matrix W for the Hopfield Network.
        W[x,i,y,j] represents the connection strength from neuron V_y,j to V_x,i.
        """
        param_A, param_B, param_D, param_C = self.params['A'], self.params['B'], self.params['D'], self.params['C']
        
        # Initialize with a base inhibitory weight from the C constraint
        W = np.full((self.n, self.n, self.n, self.n), -param_C, dtype=float)
        
        # Set self-connections to zero (neuron V_x,i does not affect itself directly)
        for r_city in range(self.n):
            for c_pos in range(self.n):
                W[r_city, c_pos, r_city, c_pos] = 0.0

        for x_idx in range(self.n): # Iterate over city indices
            for i_pos in range(self.n): # Iterate over position indices
                # Constraint A: Each city must be in only one position
                # If V_x,i is on, inhibit other V_x,j (for j != i)
                for j_other_pos in range(self.n):
                    if i_pos != j_other_pos:
                        W[x_idx, i_pos, x_idx, j_other_pos] -= param_A

                # Constraint B: Each position must have only one city
                # If V_x,i is on, inhibit other V_y,i (for y != x)
                for y_other_idx in range(self.n):
                    if x_idx != y_other_idx:
                        W[x_idx, i_pos, y_other_idx, i_pos] -= param_B
                
                # Constraint D: Minimize total path length
                # If V_x,i is on, it encourages V_y,i+1 and V_y,i-1 to be on if d_xy is small
                # This translates to negative weights proportional to distance for adjacent positions
                if param_D > 0:
                    for y_adj_idx in range(self.n):
                        if x_idx != y_adj_idx: # Avoid self-loops for distance calculation
                            # Next position in the tour
                            j_next_pos = (i_pos + 1) % self.n
                            W[x_idx, i_pos, y_adj_idx, j_next_pos] -= param_D * self.distances[x_idx, y_adj_idx]
                            # Previous position in the tour
                            j_prev_pos = (i_pos - 1 + self.n) % self.n
                            W[x_idx, i_pos, y_adj_idx, j_prev_pos] -= param_D * self.distances[x_idx, y_adj_idx]
        return W

    def _compute_bias(self):
        """
        Computes the bias matrix b for the Hopfield Network.
        This bias encourages neurons to be active.
        """
        param_A, param_B, param_C = self.params['A'], self.params['B'], self.params['C']
        # The bias term is derived from the constant part of the energy function's derivative
        # It encourages N neurons to be active in total
        return (param_A + param_B + param_C * self.n) * np.ones((self.n, self.n))

    def _compute_energy(self):
        """
        Computes the total energy of the Hopfield Network's current state.
        The network aims to minimize this energy function.
        """
        param_A, param_B, param_D, param_C = self.params['A'], self.params['B'], self.params['D'], self.params['C']
        
        # Energy term for Constraint A (each city in only one position)
        E_constraints_A = 0
        for x_city in range(self.n):
            E_constraints_A += (np.sum(self.V[x_city, :]) - 1) ** 2
        E_constraints_A *= (param_A / 2)

        # Energy term for Constraint B (each position has only one city)
        E_constraints_B = 0
        for i_pos in range(self.n):
            E_constraints_B += (np.sum(self.V[:, i_pos]) - 1) ** 2
        E_constraints_B *= (param_B / 2)

        # Energy term for Constraint D (minimize total path length)
        E_distance = 0
        if param_D > 0: # Only calculate if D is non-zero
            for x_idx in range(self.n):
                for y_idx in range(self.n):
                    if x_idx == y_idx:
                        continue # A city cannot be adjacent to itself in a tour
                    for i_pos in range(self.n):
                        v_next = self.V[y_idx, (i_pos + 1) % self.n] # Neuron for city y at next position
                        v_prev = self.V[y_idx, (i_pos - 1 + self.n) % self.n] # Neuron for city y at previous position
                        E_distance += self.distances[x_idx, y_idx] * self.V[x_idx, i_pos] * (v_next + v_prev)
            E_distance *= (param_D / 2)
        
        # Energy term for Constraint C (total number of active neurons should be N)
        E_constraints_C = 0
        if param_C > 0: # Only calculate if C is non-zero
            total_neuron_activation = np.sum(self.V)
            E_constraints_C = (param_C / 2) * (total_neuron_activation - self.n) ** 2
            
        return E_constraints_A + E_constraints_B + E_distance + E_constraints_C

    def _sigmoid(self, u, T_param):
        """
        The sigmoid activation function, incorporating the temperature parameter.
        At high temperatures, the output is closer to 0.5 (more uncertain).
        At low temperatures, the output is closer to 0 or 1 (more certain).
        """
        if abs(T_param) < 1e-10: # Handle very small temperatures to prevent division by zero
            return 1.0 if u > 0 else (0.0 if u < 0 else 0.5)
        exponent = -u / T_param
        # Clip exponent to avoid overflow in np.exp for very large/small values
        clipped_exponent = np.clip(exponent, -700, 700) 
        return 1 / (1 + np.exp(clipped_exponent))

    def _update_neurons(self, T_param):
        """
        Asynchronously updates a single randomly selected neuron.
        This includes the influence of other neurons, bias, and noise.
        """
        # Randomly select a neuron (city_index, position_index) to update
        x_rand = np.random.randint(0, self.n)
        i_rand = np.random.randint(0, self.n)
        
        # Calculate the weighted sum of inputs from all other neurons
        # This is equivalent to summing W[x,i,y,j] * V[y,j] over all y,j
        weighted_sum_inputs = np.sum(self.W[x_rand, i_rand, :, :] * self.V)
        
        # Introduce Gaussian noise, scaled by the current temperature.
        # This is a core component of Simulated Annealing, allowing the network
        # to escape local minima at higher temperatures.
        current_noise = np.random.normal(0, self.params['random_factor'] * T_param)
        
        # Calculate the total input to the selected neuron
        u_xi = weighted_sum_inputs + self.b[x_rand, i_rand] + current_noise
        
        # Update the neuron's activation using the sigmoid function
        self.V[x_rand, i_rand] = self._sigmoid(u_xi, T_param)

    def extract_path(self):
        """
        Extracts a TSP path from the continuous neuron activation matrix V.
        The city with the highest activation at each position is selected.
        Returns None if the extracted path is not valid (e.g., duplicated cities).
        """
        # For each position (column), find the city (row) with the maximum activation
        path = np.argmax(self.V, axis=0) 
        # Check if the extracted path contains all unique cities and has the correct length
        if len(set(path)) == self.n and len(path) == self.n:
            return list(path)
        return None

    def is_valid_path(self, path_nodes):
        """
        Checks if a given list of city nodes represents a valid TSP path.
        A path is valid if it's not None, has the correct number of cities,
        and all cities in the path are unique.
        """
        if path_nodes is None or len(path_nodes) != self.n:
            return False
        return len(set(path_nodes)) == self.n

    def compute_path_length(self, path_nodes):
        """
        Computes the total length of a given TSP path.
        Returns infinity if the path is not valid.
        """
        if not self.is_valid_path(path_nodes):
            return float('inf')
        
        current_calculated_length = 0
        # Sum distances between consecutive cities in the path, including return to start
        for i in range(self.n): 
            current_calculated_length += self.distances[path_nodes[i], path_nodes[(i + 1) % self.n]]
        return current_calculated_length

    def solve(self):
        """
        Runs the Hopfield Network simulation.
        If simulated annealing is enabled, temperature will decrease; otherwise, it remains constant.
        It iteratively updates neurons and tracks the best valid path found.
        """
        T = self.params['T_initial']
        # Determine how often to print status updates
        # print_interval = max(1, self.params['max_iterations'] // 100) # Removed for less verbose output

        # Reset best_path_hopfield and best_length_hopfield for each call to solve()
        self.best_path_hopfield = None
        self.best_length_hopfield = float('inf')

        for iter_count in range(self.params['max_iterations']):
            self._update_neurons(T) # Update a single neuron
            
            # Periodically check for best path (less frequent checks for performance in loops)
            # if iter_count % print_interval == 0 or iter_count == self.params['max_iterations'] - 1: # Removed for less verbose output
            current_path_hopfield = self.extract_path() # Try to extract a path
            current_length_hopfield = self.compute_path_length(current_path_hopfield) # Compute its length
            
            if self.is_valid_path(current_path_hopfield):
                if current_length_hopfield < self.best_length_hopfield:
                    self.best_length_hopfield = current_length_hopfield
                    self.best_path_hopfield = current_path_hopfield
            
            # Decrease temperature only if simulated annealing is enabled
            if self.use_simulated_annealing:
                T = max(T * self.params['alpha'], self.params['T_final'])
            # Else, T remains T_initial

        # Final check after all iterations to ensure the very last state is considered
        final_path_hopfield = self.extract_path()
        final_length_hopfield = self.compute_path_length(final_path_hopfield)
        if self.is_valid_path(final_path_hopfield) and final_length_hopfield < self.best_length_hopfield:
            self.best_path_hopfield = final_path_hopfield
            self.best_length_hopfield = final_length_hopfield
        
        return self.V, self.best_path_hopfield, self.best_length_hopfield

    def plot_path(self, path_nodes, path_len_plot, title_prefix="TSP Path"):
        """
        Plots a given TSP path on a 2D coordinate system.
        """
        if not self.is_valid_path(path_nodes): 
            print(f"Cannot plot: Invalid path for {title_prefix} (expected {self.n} unique cities). Path: {path_nodes}")
            return
        
        num_cities_in_path = self.n 

        plt.figure(figsize=(8, 8))
        plt.scatter(self.city_coords[:, 0], self.city_coords[:, 1], c='red', s=100, label='Cities')
        for i, coord in enumerate(self.city_coords):
            # Adjust text offset for better readability based on coordinate scale
            offset_scale = 0.01 * (np.max(self.city_coords[:,0]) - np.min(self.city_coords[:,0]) + 
                                   np.max(self.city_coords[:,1]) - np.min(self.city_coords[:,1])) / 2
            plt.text(coord[0] + offset_scale, 
                     coord[1] + offset_scale, 
                     f'{i}', fontsize=12) # Display city index
        
        # Plot the path segments
        for i in range(num_cities_in_path):
            start_node_idx = path_nodes[i]
            end_node_idx = path_nodes[(i + 1) % num_cities_in_path] # Connect back to the start
            start_coord = self.city_coords[start_node_idx]
            end_coord = self.city_coords[end_node_idx]
            plt.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]], 
                     'b-', lw=1.5, label='Path' if i == 0 else "") # Label only once for legend
                    
        plt.title(f"{title_prefix}, Length: {path_len_plot:.2f}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        if num_cities_in_path > 0 : # Only show legend if there's a path
            plt.legend()
        plt.grid(True)
        # Sanitize title_prefix for filename (remove special chars)
        filename_prefix = "".join(c if c.isalnum() else "_" for c in title_prefix)
        plt.savefig(f"{filename_prefix}_path.png")
        plt.close() # Close plot to prevent memory issues in loops

# --- Auxiliary function for brute-force comparison (outside the class as it's general) ---
def get_all_tours_and_lengths(num_cities, distance_matrix):
    """
    Generates all possible TSP tours and their lengths using brute force.
    This is used for verification against the optimal solution for small N.
    """
    if num_cities == 0:
        return [], float('inf'), None

    cities = list(range(num_cities))
    
    if num_cities == 1:
        return [(tuple([cities[0]]), 0.0)], 0.0, tuple([cities[0]])

    fixed_start_city = cities[0]
    other_cities = cities[1:]

    min_length = float('inf')
    best_tour = None
    all_tours_lengths = []

    for p in permutations(other_cities):
        current_tour_nodes = (fixed_start_city,) + p # Construct a full tour starting from the fixed city
        current_length = 0
        for i in range(num_cities):
            u = current_tour_nodes[i]
            v = current_tour_nodes[(i + 1) % num_cities] # Connect back to the starting city
            current_length += distance_matrix[u, v]

        all_tours_lengths.append({'tour': current_tour_nodes, 'length': current_length})
        if current_length < min_length:
            min_length = current_length
            best_tour = current_tour_nodes

    return all_tours_lengths, min_length, best_tour

def run_experiment(num_cities, city_coords, custom_params, use_simulated_annealing, num_runs, experiment_name):
    """
    Runs a series of experiments with the Hopfield TSP solver and logs the results to a file.
    Only records information for successful runs (i.e., when a valid path is found).
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tsp_experiment_results_{experiment_name}_{timestamp}.txt"

    total_valid_paths = 0
    total_optimal_paths = 0
    all_found_lengths = []
    
    # Calculate true optimal path once for comparison
    temp_solver = HopfieldTSPSolver(num_cities=num_cities, city_coords=city_coords, params=custom_params, use_simulated_annealing=use_simulated_annealing)
    _, true_min_len, true_best_tour = get_all_tours_and_lengths(num_cities, temp_solver.distances)

    with open(filename, 'w') as f:
        f.write(f"--- TSP Hopfield Network Experiment Results ---\n")
        f.write(f"Experiment Name: {experiment_name}\n")
        f.write(f"Date and Time: {timestamp}\n")
        f.write(f"Number of Cities (N): {num_cities}\n")
        f.write(f"City Coordinates:\n{city_coords}\n")
        f.write(f"Use Simulated Annealing: {use_simulated_annealing}\n")
        f.write(f"Number of Runs per Parameter Set: {num_runs}\n")
        f.write(f"Hopfield Network Parameters:\n")
        for param, value in custom_params.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"True Optimal Path Length (Brute Force): {true_min_len:.2f}\n")
        f.write(f"True Optimal Tour (Brute Force): {true_best_tour}\n")
        f.write("-" * 50 + "\n\n")

        for run_idx in range(num_runs):
            # print(f"Running {experiment_name}, Run {run_idx + 1}/{num_runs}...") # Uncomment for console progress
            tsp_solver = HopfieldTSPSolver(num_cities=num_cities, city_coords=city_coords, params=custom_params, use_simulated_annealing=use_simulated_annealing)
            final_V, hopfield_path, hopfield_length = tsp_solver.solve()

            if hopfield_path is not None:
                total_valid_paths += 1
                all_found_lengths.append(hopfield_length)
                
                f.write(f"--- Successful Run {total_valid_paths} (Overall Run {run_idx + 1})---\n")
                f.write(f"  Path Found: {hopfield_path}\n")
                f.write(f"  Path Length: {hopfield_length:.2f}\n")

                is_optimal = False
                if abs(hopfield_length - true_min_len) < 1e-5:
                    is_optimal = True
                    total_optimal_paths += 1
                f.write(f"  Is Optimal Path: {is_optimal}\n")
                f.write("\n")
            # else: # Uncomment if you want to explicitly state failed runs in the log
            #     f.write(f"--- Run {run_idx + 1} Failed ---\n")
            #     f.write(f"  No valid path found in this run.\n\n")

        f.write("-" * 50 + "\n")
        f.write(f"--- Summary for Experiment '{experiment_name}' ---\n")
        f.write(f"Total Runs: {num_runs}\n")
        f.write(f"Number of Valid Paths Found: {total_valid_paths}\n")
        f.write(f"Percentage of Valid Paths: {(total_valid_paths / num_runs * 100):.2f}%\n")
        f.write(f"Number of Optimal Paths Found: {total_optimal_paths}\n")
        if total_valid_paths > 0:
            f.write(f"Percentage of Optimal Paths (among valid): {(total_optimal_paths / total_valid_paths * 100):.2f}%\n")
            f.write(f"Average Length of Valid Paths: {np.mean(all_found_lengths):.2f}\n")
            f.write(f"Min Length of Valid Paths: {np.min(all_found_lengths):.2f}\n")
            f.write(f"Max Length of Valid Paths: {np.max(all_found_lengths):.2f}\n")
        else:
            f.write("No valid paths were found in any run.\n")
        f.write("-" * 50 + "\n")

    print(f"Experiment results saved to {filename}")
    
    # Plot brute force optimal path once at the end
    if true_best_tour:
        temp_solver.plot_path(list(true_best_tour), true_min_len, title_prefix=f"{experiment_name}_Brute_Force_Optimal_TSP_Path")


# --- Main execution block ---
if __name__ == "__main__":
    #np.random.seed(42)  # Set random seed for reproducibility of city coordinates

    n = 8  # Number of cities
    # Generate random (x, y) coordinates for the cities
    new_city_coords = np.random.rand(n, 2) * 100
    
    num_runs_per_param_set = 100 # Number of times to run each parameter set

    # --- Experiment 1: With Simulated Annealing ---
    use_sa_exp1 = True
    params_exp1 = {
        'A': 5000,
        'B': 5000,
        'D': 10,
        'C': 5000 / n,
        'T_initial': 1.0,
        'T_final': 0.01,
        'alpha': 0.99995,
        'max_iterations': 150000,
        'random_factor': 50
    }
    run_experiment(n, new_city_coords, params_exp1, use_sa_exp1, num_runs_per_param_set, "SA_Standard_Params")

    # --- Experiment 2: Without Simulated Annealing (constant T_initial, no cooling) ---
    use_sa_exp2 = False
    params_exp2 = {
        'A': 5000,
        'B': 5000,
        'D': 10,
        'C': 5000 / n,
        'T_initial': 0.1, # Keep T low and constant for no annealing
        'T_final': 0.1,
        'alpha': 1.0, # Alpha = 1.0 means no cooling (T * 1.0 = T)
        'max_iterations': 150000,
        'random_factor': 50
    }
    run_experiment(n, new_city_coords, params_exp2, use_sa_exp2, num_runs_per_param_set, "No_SA_Constant_T")
'''
    # --- Experiment 3: With Simulated Annealing, different alpha ---
    use_sa_exp3 = True
    params_exp3 = {
        'A': 500,
        'B': 500,
        'D': 10,
        'C': 500 / n,
        'T_initial': 1.0,
        'T_final': 0.01,
        'alpha': 0.999, # Faster cooling
        'max_iterations': 150000,
        'random_factor': 50
    }
    run_experiment(n, new_city_coords, params_exp3, use_sa_exp3, num_runs_per_param_set, "SA_Faster_Cooling")
'''
    # 你可以在此處添加更多實驗區塊