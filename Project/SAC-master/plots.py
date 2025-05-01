import numpy as np
import matplotlib.pyplot as plt
from config import RESULTS_DIR, PLOTS_DIR
import json
import re

def add_algorithm_results_to_plot(system_name, algorithm, epsd_steps, color='blue'):
    # Load JSON results
    with (RESULTS_DIR / f"mean_rewards_{system_name}_{epsd_steps}_5.json").open('r') as f:
        rewards = json.load(f)

    # Convert to numpy array for easier calculations
    data = np.array(rewards)

    # Calculate mean, min, and max across the 5 runs for each time step
    mean_values = np.mean(data, axis=0)
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)

    # Create time steps
    time_steps = np.arange(len(mean_values))

    # Add to the plot
    plt.plot(time_steps * epsd_steps, mean_values, color=color, label=algorithm)
    plt.fill_between(time_steps * epsd_steps, min_values, max_values, color=color, alpha=0.2)

system_name = "Pendulum-v1"
plt.figure(figsize=(10, 6))
add_algorithm_results_to_plot(system_name, "SAC", 200, color='blue')

# Add plot labels and title
plt.xlabel('Time steps')
plt.ylabel('Episode average reward')
plt.title(system_name)
plt.legend()

# Save the plot
plt.savefig(PLOTS_DIR / f'{system_name}_rewards_plot.png', dpi=300, bbox_inches='tight')
plt.close()