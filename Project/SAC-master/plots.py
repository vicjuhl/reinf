import numpy as np
import matplotlib.pyplot as plt
from config import RESULTS_DIR, PLOTS_DIR
import json
import re

def add_algorithm_results_to_plot(system_name, alg, epsd_steps):
    # Load JSON results
    with (RESULTS_DIR / f"results_{system_name}_{alg}_{epsd_steps}_5.json").open('r') as f:
        results = json.load(f)

    n_proc = len(results)
    total_rewards = [[] for _ in range(n_proc)]
    epsd_steps_actual = [[] for _ in range(n_proc)]
    for proc_id, proc_results in enumerate(results):
        for epsd_results in proc_results:
            total_rewards[proc_id].append(epsd_results["epsd_total_reward"])
            epsd_steps_actual[proc_id].append(epsd_results["final_step"] + 1)

    # Calculate mean, min, and max across the 5 runs for each time step
    # mean_values = np.mean(rewards, axis=0)
    # min_values = np.min(rewards, axis=0)
    # max_values = np.max(rewards, axis=0)

    # Create time steps
    # time_steps = np.arange(len(mean_values))

    # Add to the plot
    for proc_id in range(len(results)):
        r = total_rewards[proc_id]
        es = epsd_steps_actual[proc_id]
        acc_steps = np.cumsum(es)
        plt.plot(acc_steps, r, label=alg)
        # plt.fill_between(time_steps * epsd_steps, min_values, max_values, alpha=0.2)

def plot_system(system_name, algs, epsd_steps):
    plt.figure(figsize=(10, 6))
    for alg in algs:
        add_algorithm_results_to_plot(system_name, alg, epsd_steps)

    # Add plot labels and title
    plt.xlabel('Time steps')
    plt.ylabel('Episode average reward')
    plt.title(system_name)
    plt.legend()

    # Save the plot
    plt.savefig(PLOTS_DIR / f'{system_name}_rewards_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# plot_system("Pendulum-v1", ["SAC", "SAC2"], 200)
plot_system("Hopper-v4", ["SAC"], 1000)