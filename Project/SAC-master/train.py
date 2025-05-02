import multiprocessing as mp
import json
import pickle
from system import System  # wherever your System class lives
from pathlib import Path
from config import MODELS_DIR, RESULTS_DIR, VIDEOS_DIR, REWARD_SCALE, PLOTS_DIR, EPISODE_LENGTH, TOTAL_STEPS

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

def run_env(proc_id, alg, system_type, total_steps, epsd_steps, result_queue):
    system = System(
        system=system_type,
        alg=alg,
        reward_scale=REWARD_SCALE[system_type],
        epsd_steps=epsd_steps,
        video_freq=None,#total_steps // 5, # TODO: will this begin videos in the middle of episodes?
        proc_id=proc_id
    )
    results = system.train_agent(total_steps)
    # Save model to models directory
    model_path = MODELS_DIR / f"{system_type}_{alg}_{proc_id}.p"
    pickle.dump(system.agent, open(model_path, 'wb'))
    print(f"[Proc {proc_id}] Finished training", flush=True)
    result_queue.put((proc_id, results))
    print(f"[Proc {proc_id}] Results put to queue", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn")

    n_test = 5
    alg = 'SAC'
    # system_type = 'Pendulum-v1'
    system_type = 'Hopper-v4'
    epsd_steps = EPISODE_LENGTH[system_type]
    total_steps = TOTAL_STEPS[system_type]
    result_queue = mp.Queue()
    processes = []

    for proc_id in range(n_test):
        p = mp.Process(
            target=run_env,
            args=(proc_id, alg, system_type, total_steps, epsd_steps, result_queue)
        )
        p.start()
        processes.append(p)

    # Collect and save rewards as they arrive
    all_results = [None] * n_test
    for _ in range(n_test):
        proc_id, results = result_queue.get()  # Will block until a next result is available
        all_results[proc_id] = results

    # Save to JSON in results directory
    results_path = RESULTS_DIR / f'results_{system_type}_{alg}_{int(total_steps)}_{n_test}.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
        print("Written results to json (safe to interrupt if needed)")
        
    for p in processes:
        p.join()
    print("Joined processes successfully")
