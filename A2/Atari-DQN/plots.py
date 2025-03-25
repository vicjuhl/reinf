import matplotlib.pyplot as plt
import json
import os

def create_plots(log_dir):
    # Load the JSON data
    with open(os.path.join(log_dir, "results.json"), 'r') as f:
        results = json.load(f)

    # Extract data
    lossList = results['losses']
    rewardList = results['rewards']
    avglosslist = results['avg_losses']
    avgrewardlist = results['avg_rewards']

    # Plot loss-epoch
    plt.figure(1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(len(lossList)), lossList, label="loss")
    plt.plot(range(len(lossList)), avglosslist, label="avg")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss.png"))

    # Plot reward-epoch
    plt.figure(2)
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.plot(range(len(rewardList)), rewardList, label="reward")
    plt.plot(range(len(rewardList)), avgrewardlist, label="avg")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "reward.png"))