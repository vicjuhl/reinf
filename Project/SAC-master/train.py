import multiprocessing as mp
import json
import pickle
import argparse
from system import System  # wherever your System class lives
from pathlib import Path
from config import MODELS_DIR, RESULTS_DIR, VIDEOS_DIR, REWARD_SCALE, PLOTS_DIR, EPISODE_LENGTH, TOTAL_STEPS

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train SAC agent on different environments')
    parser.add_argument('--n_test', type=int, default=5, help='Number of parallel training runs')
    parser.add_argument('--alg', type=str, choices=['SAC', 'SACGAE'], help='Algorithm to use')
    parser.add_argument('--system_type', type=str,
                       choices=['Hopper-v4', 'Pendulum-v1', 'HalfCheetah-v4', 'Ant-v4'], help='Environment to train on')
    parser.add_argument('--total_steps', type=int, default=None, 
                       help='Total number of training steps. If None, uses default from config')
    parser.add_argument('--reward_scale', type=float, default=None,
                       help='Reward scaling factor. If None, uses default from config')
    parser.add_argument('--punishment', type=float, default=-10,
                       help='Punishment value for termination. Default is -10')
    parser.add_argument('--GAE', action='store_true', help='Enable GAE')
    parser.add_argument('--grad_steps', type=int, default=1, help='Gradient steps')
    parser.add_argument('--IS', action='store_true',help='Use importance sampling')
    return parser.parse_args()

def run_env(proc_id, alg, system_type, reward_scale, punishment, total_steps, epsd_steps, result_queue, GAE, IS):
    system = System(
        system=system_type,
        alg=alg,
        reward_scale=reward_scale,
        punishment=punishment,
        epsd_steps=epsd_steps,
        video_freq=None,#total_steps // 5, # TODO: will this begin videos in the middle of episodes?
        memory_capacity=1000,
        proc_id=proc_id,
        GAE = GAE,
        IS = IS,
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

    args = parse_args()
    n_test = args.n_test
    alg = args.alg
    system_type = args.system_type
    reward_scale = args.reward_scale if args.reward_scale is not None else REWARD_SCALE[system_type]
    epsd_steps = EPISODE_LENGTH[system_type]
    total_steps = args.total_steps if args.total_steps is not None else TOTAL_STEPS[system_type]
    punishment = args.punishment
    result_queue = mp.Queue()
    processes = []
    GAE = args.GAE
    IS = args.IS

    for proc_id in range(n_test):
        p = mp.Process(
            target=run_env,
            args=(proc_id, alg, system_type, reward_scale, punishment, total_steps, epsd_steps, result_queue, GAE, IS)
        )
        p.start()
        processes.append(p)

    # Collect and save rewards as they arrive
    all_results = [None] * n_test
    for _ in range(n_test):
        proc_id, results = result_queue.get()  # Will block until a next result is available
        all_results[proc_id] = results

    # Save to JSON in results directory
    results_path = RESULTS_DIR / f'results_{system_type}_{alg}_{int(total_steps)}_{n_test}_{int(reward_scale)}_{int(punishment)}.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
        print("Written results to json (safe to interrupt if needed)")
        
    for p in processes:
        p.join()
    print("Joined processes successfully")
