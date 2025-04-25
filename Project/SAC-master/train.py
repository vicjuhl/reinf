import multiprocessing as mp
import json
import pickle
from system import System  # wherever your System class lives
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
MODELS_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def run_env(proc_id, system_type, tr_epsds, epsd_steps, result_queue):
    system = System(system=system_type, reward_scale=100)
    rewards = system.train_agent(tr_epsds, epsd_steps)
    # Save model to models directory
    model_path = MODELS_DIR / f"{system_type}_{proc_id}.p"
    pickle.dump(system, open(model_path, 'wb'))
    result_queue.put((proc_id, rewards))

if __name__ == "__main__":
    mp.set_start_method("spawn")

    n_test = 5
    system_type = 'Hopper-v4'
    tr_epsds = 3
    epsd_steps = 50

    result_queue = mp.Queue()
    processes = []

    for proc_id in range(n_test):
        p = mp.Process(
            target=run_env,
            args=(proc_id, system_type, tr_epsds, epsd_steps, result_queue)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Collect and save rewards
    all_rewards = [None] * n_test
    while not result_queue.empty():
        proc_id, rewards = result_queue.get()
        all_rewards[proc_id] = rewards

    # Convert NumPy arrays inside to lists (if any)
    all_rewards_clean = [[float(r) for r in episode] for episode in all_rewards]
    # Save to JSON in results directory
    results_path = RESULTS_DIR / f'mean_rewards_{system_type}_{n_test}.json'
    with open(results_path, 'w') as f:
        json.dump(all_rewards_clean, f, indent=2)