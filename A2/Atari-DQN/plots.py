import matplotlib.pyplot as plt
import json
import os
import pandas as pd

def create_plots(log_dir):
    # Load the JSON data for training metrics
    with open(os.path.join(log_dir, "results.json"), 'r') as f:
        results = json.load(f)

    # Load evaluation data from CSV
    eval_df = pd.read_csv(os.path.join(log_dir, "eval_rewards.csv"))
    
    # Calculate mean rewards per epoch (in case there are multiple evaluations per epoch)
    eval_means = eval_df.groupby('epoch')['reward'].mean().reset_index()
    
    # Extract data
    lossList = results['losses']
    rewardList = results['rewards']
    avglosslist = results['avg_losses']
    avgrewardlist = results['avg_rewards']

    # Plot loss-epoch
    plt.figure(1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(len(lossList)), lossList, alpha=0.3, label="Training Loss")
    plt.plot(range(len(lossList)), avglosslist, label="Moving Average")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss.png"))
    plt.close()

    # Plot reward-epoch
    plt.figure(2)
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    # Training rewards (more transparent)
    plt.plot(range(len(rewardList)), rewardList, alpha=0.3, label="Training Reward")
    plt.plot(range(len(rewardList)), avgrewardlist, label="Training Moving Average")
    # Evaluation rewards
    plt.plot(eval_means['epoch'], eval_means['reward'], 'r-', label="Evaluation Reward")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "reward.png"))
    plt.close()
