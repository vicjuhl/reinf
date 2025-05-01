import multiprocessing as mp
import json
import pickle
from system import System  # wherever your System class lives
from pathlib import Path
from config import MODELS_DIR, RESULTS_DIR, VIDEOS_DIR, REWARD_SCALE, PLOTS_DIR

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

def run_env(proc_id, alg, system_type, tr_epsds, epsd_steps, result_queue):
    system = System(
        system=system_type,
        alg=alg,
        reward_scale=REWARD_SCALE[system_type],
        epsd_steps=epsd_steps,
        video_freq=(tr_epsds // (5)),
        proc_id=proc_id
    )
    rewards = system.train_agent(tr_epsds)
    # Save model to models directory
    model_path = MODELS_DIR / f"{system_type}_{alg}_{proc_id}.p"
    pickle.dump(system.agent, open(model_path, 'wb'))
    result_queue.put((proc_id, rewards))

if __name__ == "__main__":
    mp.set_start_method("spawn")

    n_test = 5
    alg = 'SAC2'
    system_type = 'Pendulum-v1'
    tr_epsds = 10
    epsd_steps = 200
    result_queue = mp.Queue()
    processes = []

    for proc_id in range(n_test):
        p = mp.Process(
            target=run_env,
            args=(proc_id, alg, system_type, tr_epsds, epsd_steps, result_queue)
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
    results_path = RESULTS_DIR / f'mean_rewards_{system_type}_{alg}_{epsd_steps}_{n_test}.json'
    with open(results_path, 'w') as f:
        json.dump(all_rewards_clean, f, indent=2)