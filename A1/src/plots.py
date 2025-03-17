import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np

def plot_total_rewards_subplot(river_length, r_total_stack, π_star_learned_stack, n_iter_stack, ax):
    """Plot total rewards at the end of each episode on a given subplot."""
    # Longest run length among experiments
    longest = np.max(n_iter_stack)
    
    # Get final rewards for each episode
    final_rewards = r_total_stack[:, :longest, -1].copy()

    # Replace uninitialized (-1) entries with the average of last few episodes for that run (vectorized)
    mask = final_rewards == -1
    for i in range(final_rewards.shape[0]):
        n = n_iter_stack[i]
        avg_last_few = final_rewards[i, n-4:n+1].mean()
        final_rewards[i, mask[i, :]] = avg_last_few
        
    # Calculate mean across runs, excluding -1 values
    avg_final_rewards = final_rewards[:, :longest].mean(axis=0)
    
    # Get standard matplotlib colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Plot the average rewards in black
    ax.plot(
        avg_final_rewards,
        '-o',
        markersize=5,
        color="black"
    )

    for i in range(len(n_iter_stack)):
        # Plot the rewards with the corresponding color
        last = n_iter_stack[i]
        ax.plot(
            final_rewards[i, :last+1],
            '-o',
            markersize=4,
            color=colors[i],
            alpha=0.5
        )
        
        # If optimal policy was found, mark the point with the same color
        if π_star_learned_stack[i] != -1:
            ax.axvline(
                x=π_star_learned_stack[i],
                linestyle='--',
                label=f'Optimal Policy Found (Run {i+1})',
                color=colors[i],
                alpha=0.5
            )
    
    # Labels and title
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(f'River Length {river_length}')

def visualize(results):
    """Create a figure with 2x2 subplots"""
    _, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs = axs.ravel()  # Flatten axs to make it easier to iterate
    
    # Stack results in separate arrays
    for idx, (river_length, res) in enumerate(results.items()):
        r_total_stack, π_star_learned_stack, n_iter_stack = map(np.array, zip(*res))
        
        # Plot on the corresponding subplot
        plot_total_rewards_subplot(
            river_length, 
            r_total_stack, 
            π_star_learned_stack, 
            n_iter_stack, 
            ax=axs[idx]
        )
    
    plt.tight_layout()
    plt.savefig('learning_curve.png')