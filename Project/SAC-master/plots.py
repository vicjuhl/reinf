import numpy as np
import matplotlib.pyplot as plt
from config import RESULTS_DIR, PLOTS_DIR
import json
import re
from pathlib import Path
import glob

def find_results_file(system_name, alg, total_steps):
    pattern = f"results_{system_name}_{alg}_{total_steps}*.json"
    matching_files = list(RESULTS_DIR.glob(pattern))
    if not matching_files:
        raise FileNotFoundError(f"No matching results file found for pattern: {pattern}")
    return matching_files[0]

def add_alg_results_to_plot(system_name, alg, total_steps):
    # Load JSON results
    results_file = find_results_file(system_name, alg, total_steps)
    with results_file.open('r') as f:
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

    # time_steps = np.arange(len(mean_values))

    for proc_id in range(len(results)):
        r = total_rewards[proc_id]
        es = epsd_steps_actual[proc_id]
        acc_steps = np.cumsum(es)
        plt.plot(acc_steps, r, linewidth=0.5, alpha=0.5)
        # plt.fill_between(time_steps * epsd_steps, min_values, max_values, alpha=0.2)

def add_epsd_len_to_plot(system_name, alg, total_steps):
    # Load JSON results
    results_file = find_results_file(system_name, alg, total_steps)
    with results_file.open('r') as f:
        results = json.load(f)

    n_proc = len(results)
    epsd_steps_actual = [[] for _ in range(n_proc)]
    for proc_id, proc_results in enumerate(results):
        for epsd_results in proc_results:
            epsd_steps_actual[proc_id].append(epsd_results["final_step"] + 1)

    for proc_id in range(len(results)):
        plt.plot(epsd_steps_actual[proc_id], label=alg)

def plot_system(system_name, algs, total_steps):
    # REWARDS
    plt.figure(figsize=(10, 6))
    for alg in algs:
        add_alg_results_to_plot(system_name, alg, total_steps)

    # Add plot labels and title
    plt.xlabel('Time steps')
    plt.ylabel('Episode total reward')
    plt.title(system_name)
    plt.legend()

    # Save the plot
    plt.savefig(PLOTS_DIR / f'{system_name}_rewards_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # EPISODE LENGTH
    plt.figure(figsize=(10, 6))
    for alg in algs:
        add_epsd_len_to_plot(system_name, alg, total_steps)
    
    # Add plot labels and title
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.title(system_name)
    plt.legend()

    # Save the plot
    plt.savefig(PLOTS_DIR / f'{system_name}_epsd_len_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# plot_system("Pendulum-v1", ["SAC", "SAC2"], 200)
plot_system("Hopper-v4", ["SAC"], int(1e6))
plot_system("HalfCheetah-v4", ["SAC"], int(1e6))
plot_system("Ant-v4", ["SAC"], int(1e6))