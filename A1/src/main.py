import argparse
from river import simulate
from plots import visualize
import json
import os
import numpy as np
from multiprocessing import Pool

def dump_json(results):
    """Save simulation results to JSON file."""
    # Ensure src directory exists
    os.makedirs('src', exist_ok=True)
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        str(river_length): [
            (r_total.tolist(), int(π_star), int(n_iter))
            for r_total, π_star, n_iter in runs
        ]
        for river_length, runs in results.items()
    }
    # Write results to JSON file
    with open('src/results.json', 'w') as f:
        json.dump(json_results, f)

def load_json():
    """Load simulation results from JSON file."""
    with open('src/results.json', 'r') as f:
        json_results = json.load(f)
    
    # Convert JSON data back to original format
    return {
        int(river_length): [
            (np.array(r_total), π_star, n_iter)
            for r_total, π_star, n_iter in runs
        ]
        for river_length, runs in json_results.items()
    }

def parse_arguments():
    parser = argparse.ArgumentParser(description='River crossing simulation')
    parser.add_argument('-r_pol', '--random_policy', type=float, required=True,
                        help='Randomness level when defining the deterministic policy')
    parser.add_argument('-r_act', '--random_action', type=float, required=True,
                        help='Randomness level when realizing the actions')
    parser.add_argument('-T', '--period', type=int, required=True,
                        help='Period for each simulation')
    parser.add_argument('-k', '--episodes', type=int, required=True,
                        help='Maximal number of episodes to run')
    parser.add_argument('-m', '--exploration_patience', type=int, required=True,
                        help='Number of (s,a) realizations needed to explore pair')
    parser.add_argument('-R', '--r_max', type=float, required=True,
                        help='Maximum reward value')
    parser.add_argument('-g', '--gamma', type=float, required=True,
                        help='Discount factor')
    parser.add_argument('-s', '--simulate', action='store_true',
                        help='Whether to run simulation')
    return parser.parse_args()

def run_simulation(river_length, k, period, γ, m, r_max, rand_pol, rand_act):
    print("Running simulation with river length", river_length)
    return simulate(k, period, river_length, γ, m, r_max, rand_pol, rand_act)

def main():
    args = parse_arguments()
    rand_pol = args.random_policy
    rand_act = args.random_action
    period = int(args.period)
    k = int(args.episodes)
    m = int(args.exploration_patience)
    r_max = float(args.r_max)
    γ = float(args.gamma)
    sim = bool(args.simulate)

    if rand_act < 0 or rand_act > 1:
        print("rand_act must be in [0, 1]")
        return 1
    
    if sim:
        with Pool(5) as pool:
            river_lengths = [5, 10, 15, 20]
            results = {}
            
            for length in river_lengths:
                print(f"Running 5 parallel simulations for river length {length}...")
                results[length] = pool.starmap(
                    run_simulation,
                    [(length, k, period, γ, m, r_max, rand_pol, rand_act) for _ in range(5)]
                )
                print(f"Completed simulations for river length {length}")
            
        dump_json(results)
    
    results = load_json()
    visualize(results)
    return 0

if __name__ == "__main__":
    main()
