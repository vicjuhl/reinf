import gymnasium as gym
import argparse
from model import DQN, DuelDQN
from torch import optim
from utils import Transition, ReplayMemory, VideoRecorder
from wrapper import AtariWrapper
import numpy as np
import random
import torch
import torch.nn as nn
from itertools import count
import os
import math
from collections import deque
import time
import psutil
import gc
import json
import sys
from plots import create_plots
import pandas as pd
from statistics import mean


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024  # Convert to MB
    return mem

def print_tensor_sizes():
    """Print sizes of all tensors in memory"""
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                size = obj.element_size() * obj.nelement() / 1024 / 1024  # Size in MB
                total_size += size
                # print(f"Tensor of shape {obj.shape} and type {obj.dtype}: {size:.2f} MB")
        except:
            pass
    print(f"Total tensor memory: {total_size:.2f} MB")

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--env-name',default="breakout",type=str,choices=["pong","breakout","boxing"], help="env name")
parser.add_argument('--model', default="q", type=str, choices=["q","e-sarsa"], help="dqn model (q-learning, expected SARSA)")
parser.add_argument('--double',action='store_true', help="double dqn")
parser.add_argument('--duel',action='store_true', help="dueling dqn")
parser.add_argument('--ims',action='store_true', help="importance sampling for e-SARSA")
parser.add_argument('--lr', default=2.5e-4, type=float, help="learning rate")
parser.add_argument('--epoch', default=10001, type=int, help="training epoch")
parser.add_argument('--batch-size', default=32, type=int, help="batch size")
parser.add_argument('--eval-cycle', default=500, type=int, help="evaluation cycle")
parser.add_argument('--steps-done', type=int, help="number of steps done (from model filename)")
parser.add_argument('--exp_id', type=int, help="experiement identifier")
parser.add_argument('--sim', action='store_true', help="whether to simulate (not set goes straight to plotting)")
parser.add_argument('--smaller', action='store_true', help="make neural network smaller for computational efficiency")
args = parser.parse_args()

if args.exp_id is None or args.exp_id == 0:
    print("exp_id (non-zero positive integer) must be provided")
    sys.exit(0)

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

torch.set_default_dtype(torch.float32)

# some hyperparameters
GAMMA = 0.99 # bellman function
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 50000
WARMUP = 1000 # don't update net until WARMUP steps
N_EVALS = 5
VIDEO_CYCLE = args.eval_cycle * 10

# make model specific dir and configurations
alg_name = args.model
if args.double:
    alg_name = f"double_{alg_name}"
if args.duel:
    alg_name = f"duel_{alg_name}"
if args.model == "e-sarsa" and args.ims:
    alg_name = f"{alg_name}_importance_sampling"

log_dir = os.path.join(f"log_{args.env_name}", alg_name, f"exp_{str(args.exp_id)}")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir,"log.txt")
json_path = os.path.join(log_dir,"results.json")
# video
video = VideoRecorder(log_dir)

if not args.sim:
    create_plots(log_dir)
    sys.exit(0)

def set_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed * 1)
    np.random.seed(seed * 2)
    torch.manual_seed(seed * 3)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed * 4)
        torch.cuda.manual_seed_all(seed * 5)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed * 6)

# Set all random seeds
set_seeds(345 * args.exp_id)

# Environment
if args.env_name == "pong":
    env = gym.make("PongNoFrameskip-v4")
    evalenv = gym.make("PongNoFrameskip-v4")
elif args.env_name == "breakout":
    env = gym.make("BreakoutNoFrameskip-v4")
    evalenv = gym.make("BreakoutNoFrameskip-v4")
else:
    env = gym.make("BoxingNoFrameskip-v4")
    evalenv = gym.make("BoxingNoFrameskip-v4")

env = AtariWrapper(env)
env.unwrapped.seed(456 * args.exp_id)  # If you want to seed the base environment
env.action_space.seed(567 * args.exp_id)  # Set action space seed

evalenv = AtariWrapper(evalenv, video=video)
evalenv.unwrapped.seed(123 * args.exp_id)
evalenv.action_space.seed(234 * args.exp_id)

n_action = env.action_space.n # pong:6; breakout:4; boxing:18

# create network and target network
if not args.duel:
    policy_net = DQN(in_channels=4, n_actions=n_action, smaller=args.smaller).to(device)
    target_net = DQN(in_channels=4, n_actions=n_action, smaller=args.smaller).to(device)
else:
    policy_net = DuelDQN(in_channels=4, n_actions=n_action).to(device)
    target_net = DuelDQN(in_channels=4, n_actions=n_action).to(device)
target_net.load_state_dict(policy_net.state_dict())

# replay memory
memory = ReplayMemory(25000)

# optimizer
optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, amsgrad=True)
criterion = nn.MSELoss()

# warming up
print("Warming up...")
warmupstep = 0
for epoch in count():
    obs, info = env.reset() # (84,84)
    obs = torch.from_numpy(obs).to(device).float() #(84,84)
    # stack four frames together, hoping to learn temporal info
    obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0) #(1,4,84,84)

    p_action = torch.tensor([1 / n_action], dtype=torch.float32, device=device) # No ε yet, so always p(a|s) = 1 / |A|.
    # step loop
    for step in count():
        warmupstep += 1
        # take one step
        action = torch.tensor([[env.action_space.sample()]]).to(device)
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        # convert to tensor
        reward = torch.tensor([reward], device=device, dtype=torch.float32) # (1)
        done = torch.tensor([done], device=device) # (1)
        next_obs = torch.from_numpy(next_obs).to(device).float() # (84,84)
        next_obs = torch.stack((next_obs,obs[0][0],obs[0][1],obs[0][2])).unsqueeze(0) # (1,4,84,84)
        
        # store the transition in memory
        memory.push(obs,action,p_action,next_obs,reward,done)
        
        # move to next state
        obs = next_obs

        if done:
            break

    if warmupstep > WARMUP:
        break

rewardList = []
lossList = []
rewarddeq = deque([], maxlen=100)
lossdeq = deque([],maxlen=100)
avgrewardlist = []
avglosslist = []
eval_results = []

# Initialize steps_done
steps_done = args.steps_done if args.steps_done is not None else 0
eps_threshold = EPS_START

def p_of_a_given_s(is_greedy):
    '''
    Calculate p(a|s) where s is implicit (according to call scope) and
    a is either implicit by call scope (when is_greedy.shape is (bs,1))
    or all a options are evaluated (when is_greedy.shape is (bs,n_actions)).

    Input:
        is_greedy torch.Tensor(torch.float32) with values in {0., 1.}, shape (bs,n_actions)
        describes whether a is the current greedy choice given s.
        When dim1 == 1, is_greedy is a regulare bool tensor (formated as float32).
        When dim1 == n_actions, is_greedy is a one-hot encoding where 1 means the greedy choice.

    Output: torch.Tensor(torch.float32) array of cond. probs., shape (bs,1)
    '''
    global eps_threshold
    global n_action
    return is_greedy * (1 - eps_threshold) + eps_threshold / n_action

def select_action(state:torch.Tensor)->torch.Tensor:
    '''
    epsilon greedy
    - epsilon: choose random action
    - 1-epsilon: argmax Q(a,s)

    Input: state shape (1,4,84,84)

    Output: action shape (1,1)
    '''
    global eps_threshold
    global steps_done
    global n_action
    global device
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    with torch.no_grad():
        a_greedy = policy_net(state).max(1)[1].view(1, 1)
        # Exploit
        if sample > eps_threshold: # a == π(s)
            a = a_greedy
        # Explore
        else: # a sampled uniformly
            a = torch.tensor([[env.action_space.sample()]]).to(device)
        p_a = p_of_a_given_s(torch.tensor([a_greedy == a], dtype=torch.float32, device=device))
    return a, p_a

t_0 = time.time()

# epoch loop
for epoch in range(args.epoch):
    obs, info = env.reset() # (84,84)
    obs = torch.from_numpy(obs).to(device).float() #(84,84)
    # stack four frames together, hoping to learn temporal info
    obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0) #(1,4,84,84)

    total_loss = 0.0
    total_reward = 0

    # step loop
    for step in count():
        # take one step
        action, p_action = select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward
        done = terminated or truncated
        
        # convert to tensor
        reward = torch.tensor([reward], device=device, dtype=torch.float32) # (1)
        done = torch.tensor([done], device=device) # (1)
        next_obs = torch.from_numpy(next_obs).to(device).float() # (84,84)
        next_obs = torch.stack((next_obs,obs[0][0],obs[0][1],obs[0][2])).unsqueeze(0) # (1,4,84,84)
        
        # store the transition in memory
        memory.push(obs,action,p_action,next_obs,reward,done)
        
        # move to next state
        obs = next_obs

        # train
        policy_net.train()
        transitions = memory.sample(args.batch_size)
        batch = Transition(*zip(*transitions)) # batch-array of Transitions -> Transition of batch-arrays.
        state_batch = torch.cat(batch.state) # (bs,4,84,84)
        next_state_batch = torch.cat(batch.next_state) # (bs,4,84,84)
        action_batch = torch.cat(batch.action) # (bs,1)
        p_action_batch = torch.cat(batch.p_action) # (bs,1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1) # (bs,1)
        done_batch = torch.cat(batch.done).unsqueeze(1) #(bs,1)

        # Q(st,a)
        state_qvalues = policy_net(state_batch) # (bs,n_actions)
        selected_state_qvalue = state_qvalues.gather(1,action_batch) # (bs,1) # Q(s,a) straight from the buffer
        
        # td target
        if args.model == "q": # Q-learning
            with torch.no_grad():
                # Q'(st+1,a)
                next_state_target_qvalues = target_net(next_state_batch) # (bs,n_actions)
                if args.double:
                    # Q(st+1,a)
                    next_state_qvalues = policy_net(next_state_batch) # (bs,n_actions)
                    # argmax Q(st+1,a)
                    next_state_selected_action = next_state_qvalues.max(1,keepdim=True)[1] # (bs,1)
                    # Q'(st+1,argmax_a Q(st+1,a))
                    next_state_selected_qvalue = next_state_target_qvalues.gather(1,next_state_selected_action) # (bs,1)
                else:
                    # max_a Q'(st+1,a)
                    next_state_selected_qvalue = next_state_target_qvalues.max(1,keepdim=True)[0] # (bs,1)

            tdtarget = reward_batch + next_state_selected_qvalue * GAMMA * ~done_batch
            loss = criterion(selected_state_qvalue, tdtarget)

        elif args.model == "e-sarsa": # Expected SARSA
            with torch.no_grad():
                # Create weights w
                if args.ims:
                    # Importance sampling
                    greedy_actions = state_qvalues.max(1)[1].view(-1, 1) # Actions as chosen by current policy (bs,1)
                    is_greedy = action_batch == greedy_actions # Whether a is the greedy choice (bs,1)
                    p_new_a_given_s = p_of_a_given_s(is_greedy) # p_new(a|s) according to current policy for all a, (bs,1)
                    w = p_new_a_given_s / (p_action_batch + 0.1) # weights = p_new(a|s) / (p_old(a|s) + δ), (bs,1)
                else:
                    # Uniform importance sampling
                    w = torch.ones_like(p_action_batch, dtype=torch.float32, device=device)

                # Create target
                next_state_qvalues = target_net(next_state_batch) # Q(s',a'), (bs,n_actions)
                next_greedy_actions = next_state_qvalues.max(1)[1].view(-1, 1) # (bs,1)
                next_is_greedy_one_hot = torch.zeros_like(next_state_qvalues, device=device) # (bs,n_actions)
                next_is_greedy_one_hot.scatter_(1, next_greedy_actions, 1) # (bs,n_actions)
                next_p_new_all_a = p_of_a_given_s(next_is_greedy_one_hot) # (bs,n_actions)
                expected_q = (next_p_new_all_a * next_state_qvalues).sum(dim=1, keepdim=True) # (bs,1)
                tdtarget = reward_batch + GAMMA * expected_q * ~done_batch # (bs,1)

            # Compute loss with importance sampling
            loss_elements = criterion(selected_state_qvalue, tdtarget) # (bs,1)
            loss = (w * loss_elements).mean()

        else:
            raise ValueError(f"Unknown algorithm: {args.alg}")

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        del state_batch          # Input batch, already used
        del next_state_batch     # Only used for computing tdtarget
        del action_batch         # Only used for selecting Q-values
        del p_action_batch       # Only used for this step
        del reward_batch         # Only used for computing tdtarget
        del done_batch          # Only used for computing tdtarget
        
        # Need to be more careful with these: TODO
        del selected_state_qvalue  # Used in loss computation
        del tdtarget              # Used in loss computation

        # let target_net = policy_net every 1000 steps
        if steps_done % 1000 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            # eval
            if epoch % args.eval_cycle == 0:
                t_eval_0 = time.time()
                with torch.no_grad():
                    save_video = (epoch % VIDEO_CYCLE == 0)
                    policy_net.eval()
                    for eval_round in range(N_EVALS):
                        if save_video:
                            video.reset()
                        obs, info = evalenv.reset()
                        obs = torch.from_numpy(obs).to(device).float()
                        obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0)
                        evalreward = 0
                        for _ in count(): # Step count
                            action = policy_net(obs).max(1)[1]
                            next_obs, reward, terminated, truncated, info = evalenv.step(action.item())
                            evalreward += reward
                            next_obs = torch.from_numpy(next_obs).to(device).float() # (84,84)
                            next_obs = torch.stack((next_obs,obs[0][0],obs[0][1],obs[0][2])).unsqueeze(0) # (1,4,84,84)
                            obs = next_obs
                            if terminated or truncated:
                                if info["lives"] == 0: # real end
                                    break
                                else:
                                    obs, info = evalenv.reset()
                                    obs = torch.from_numpy(obs).to(device).float()
                                    obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0)
                        eval_results.append((epoch, evalreward))
                        if save_video:
                            video.save(f"{epoch}.mp4")
                            save_video = False
                    evalenv.close()
                    torch.save(policy_net, os.path.join(log_dir,f'model{epoch}_{steps_done}.pth'))
                    print(f"Eval epoch {epoch}: Reward {mean([r for _, r in eval_results])}")
                    print(f"time taken to evaluate was {time.time() - t_eval_0}")
                    
                    # Clear cache after evaluation
                    if device.type == "mps":
                        torch.mps.empty_cache()
            break
    
    rewardList.append(total_reward)
    lossList.append(total_loss)
    rewarddeq.append(total_reward)
    lossdeq.append(total_loss)
    avgreward = sum(rewarddeq)/len(rewarddeq)
    avgloss = sum(lossdeq)/len(lossdeq)
    avglosslist.append(avgloss)
    avgrewardlist.append(avgreward)

    output = f"Epoch {epoch}: Loss {total_loss:.2f}, Reward {total_reward}, Avgloss {avgloss:.2f}, Avgreward {avgreward:.2f}, Epsilon {eps_threshold:.2f}, TotalStep {steps_done}"
    print(output)
    with open(log_path,"a") as f:
        f.write(f"{output}\n")

    # Optionally, clear cache every N epochs
    if epoch % 100 == 0 and device.type == "mps":
        torch.mps.empty_cache()

    # if epoch % 25 == 0:  # Print every 100 epochs
    #     print(f"\nMemory diagnostics at epoch {epoch}:")
    #     print(f"Current memory usage: {get_memory_usage():.2f} MB")
    #     print("Tensor sizes in memory:")
    #     print_tensor_sizes()
    #     print(f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1024/1024:.2f} MB") if device.type == "cuda" else None
        
    # After the evaluation step:
    # if epoch % args.eval_cycle == 0:
    #     print("\nMemory before garbage collection:")
    #     print(f"Memory usage: {get_memory_usage():.2f} MB")
    #     gc.collect()
    #     if device.type == "cuda":
    #         torch.cuda.empty_cache()
    #     elif device.type == "mps":
    #         torch.mps.empty_cache()
    #     print("Memory after garbage collection:")
    #     print(f"Memory usage: {get_memory_usage():.2f} MB")

# Print time statistic
total_time = time.time() - t_0
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
print(f"\nTotal training time: {hours}h {minutes}m {seconds}s")

# Save eval reward results
df_eval_results = pd.DataFrame(eval_results, columns=['epoch', 'reward'])
df_eval_results.to_csv(os.path.join(log_dir, 'eval_rewards.csv'), index=False)

env.close()

# Save results to JSON
results = {
    'rewards': rewardList,
    'losses': lossList,
    'avg_rewards': avgrewardlist,
    'avg_losses': avglosslist
}
with open(json_path, 'w') as f:
    json.dump(results, f)

create_plots(log_dir)
