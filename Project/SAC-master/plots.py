import numpy as np
import matplotlib.pyplot as plt
from config import RESULTS_DIR, PLOTS_DIR
import json
import re

# Extract episode steps from filename using regex
filename = "mean_rewards_Pendulum-v1_200_5.json"
epsd_steps = int(re.search(r'_(\d+)_', filename).group(1))

# Load JSON results
with (RESULTS_DIR / f"mean_rewards_Pendulum-v1_{epsd_steps}_5.json").open('r') as f:
    rewards_pend_1 = json.load(f)

# Convert to numpy array for easier calculations
data = np.array(rewards_pend_1)

# Calculate mean, min, and max across the 5 runs for each time step
mean_values = np.mean(data, axis=0)
min_values = np.min(data, axis=0)
max_values = np.max(data, axis=0)

# Create time steps
time_steps = np.arange(len(mean_values))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(time_steps * epsd_steps, mean_values, color='blue')
plt.fill_between(time_steps * epsd_steps, min_values, max_values, color='blue', alpha=0.2)

# Remove all text
plt.xlabel('Time steps')
plt.ylabel('Episode average reward')
plt.title('Pendulum-v1')

# Save the plot
plt.savefig(PLOTS_DIR / 'Pendulum-v1_rewards_plot.png', dpi=300, bbox_inches='tight')
plt.close()