import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym                 #going to the gym
from nets import Memory, v_valueNet, q_valueNet, policyNet
from gymnasium.wrappers import RecordVideo, TimeLimit
from config import VIDEOS_DIR
from itertools import count

from sys import stdout
import pickle
import time

# Device configuration
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
device = "cpu"
print(f"Using device: {device}")

###########################################################################
#
#                           General functions
#
###########################################################################
def updateNet(target, source, tau):    
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )

def reset_env(self):
    obs, info = self.env.reset()
    if isinstance(obs, tuple):
        state = obs[0]  # For environments like Pendulum-v1 which return a tuple
    else:
        state = obs
    return state

def normalize_angle(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

###########################################################################
#
#                               Classes
#
###########################################################################
#-------------------------------------------------------------
#
#    SAC agent
#
#-------------------------------------------------------------
class Agent:
    '''
    Attributes:    

    Methods:
    fit --  
    s_score --  
    sample_a -- 
    sample_m_state -- 
    act --    
    learn --
    '''

    def __init__(self, s_dim=2, a_dim=1, memory_capacity=50000, batch_size=64, discount_factor=0.99, temperature=1.0,
        soft_lr=5e-3, reward_scale=1.0):
        '''
        Initializes the agent.

        Arguments:

        Returns:
        none
        '''
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.sa_dim = self.s_dim + self.a_dim          
        self.batch_size = batch_size 
        self.gamma = discount_factor
        self.soft_lr = soft_lr        
        self.alpha = temperature
        self.reward_scale = reward_scale
         
        self.memory = Memory(memory_capacity)
        self.actor = policyNet(s_dim, a_dim).to(device)        
        self.critic1 = q_valueNet(self.s_dim, self.a_dim).to(device)
        self.critic2 = q_valueNet(self.s_dim, self.a_dim).to(device)
        self.baseline = v_valueNet(s_dim).to(device) 
        self.baseline_target = v_valueNet(s_dim).to(device) 
    
        updateNet(self.baseline_target, self.baseline, 1.0) 

    def act(self, state, explore=True):
        with torch.no_grad():
            action = self.actor.sample_action(state)
            return action
    
    def memorize(self, event):
        self.memory.store(event[np.newaxis,:])
    
    def learn(self):        
        batch = self.memory.sample(self.batch_size)
        batch = np.concatenate(batch, axis=0)
       
        s_batch = torch.FloatTensor(batch[:,:self.s_dim]).to(device)
        a_batch = torch.FloatTensor(batch[:,self.s_dim:self.sa_dim]).to(device)
        r_batch = torch.FloatTensor(batch[:,self.sa_dim]).unsqueeze(1).to(device)
        ns_batch = torch.FloatTensor(batch[:,self.sa_dim+1:self.sa_dim+1+self.s_dim]).to(device)

        # Optimize q networks
        q1 = self.critic1(s_batch, a_batch)
        q2 = self.critic2(s_batch, a_batch)     
        next_v = self.baseline_target(ns_batch)
        q_approx = self.reward_scale * r_batch + self.gamma * next_v

        q1_loss = self.critic1.loss_func(q1, q_approx.detach())
        self.critic1.optimizer.zero_grad()
        q1_loss.backward()
        self.critic1.optimizer.step()
        
        q2_loss = self.critic2.loss_func(q2, q_approx.detach())
        self.critic2.optimizer.zero_grad()
        q2_loss.backward()
        self.critic2.optimizer.step()

        # Optimize v network
        v = self.baseline(s_batch)
        a_batch_off, llhood = self.actor.sample_action_and_llhood(s_batch)                
        q1_off = self.critic1(s_batch, a_batch_off)
        q2_off = self.critic2(s_batch, a_batch_off)
        q_off = torch.min(q1_off, q2_off)          
        v_approx = q_off - self.alpha*llhood

        v_loss = self.baseline.loss_func(v, v_approx.detach())
        self.baseline.optimizer.zero_grad()
        v_loss.backward()
        self.baseline.optimizer.step()
        
        # Optimize policy network
        pi_loss = (llhood - q_off).mean()
        self.actor.optimizer.zero_grad()
        pi_loss.backward()
        self.actor.optimizer.step()

        # Update v target network
        updateNet(self.baseline_target, self.baseline, self.soft_lr)

#-------------------------------------------------------------
#
#    SAC system
#
#-------------------------------------------------------------
class System:
    def __init__(self, alg, memory_capacity = 200000, env_steps=1, grad_steps=1, init_steps=256, reward_scale = 25,
        temperature=1.0, soft_lr=5e-3, batch_size=256, system='Hopper-v4',
        epsd_steps=200, video_freq=200, proc_id=-1): # 'Pendulum-v0', 'Hopper-v4', 'HalfCheetah-v2', 'Swimmer-v2'

        env = gym.make(
            system,
            render_mode="rgb_array" if video_freq is not None else None
        )
        env = TimeLimit(env, max_episode_steps=epsd_steps)
        if video_freq is not None and proc_id == 0:
            print(f"Saving videos for simulation for process id {proc_id}...")
            env = RecordVideo(env, name_prefix=f"{system}_{alg}", video_folder=VIDEOS_DIR,
                              step_trigger=lambda s: s % video_freq == 0)
        self.env = env
        self.env.reset(seed=None)
        self.type = system
       
        self.s_dim = self.env.observation_space.shape[0]
        self.a_dim = self.env.action_space.shape[0] 
        self.sa_dim = self.s_dim + self.a_dim
        self.e_dim = self.s_dim*2 + self.a_dim + 1

        self.env_steps = env_steps
        self.grad_steps = grad_steps
        self.init_steps = init_steps
        self.epsd_steps = epsd_steps
        self.batch_size = batch_size

        self.min_action = self.env.action_space.low
        self.max_action = self.env.action_space.high
        self.temperature = temperature
        self.reward_scale = reward_scale

        self.agent = Agent(
            s_dim=self.s_dim,
            a_dim=self.a_dim,
            memory_capacity=memory_capacity,
            batch_size=batch_size,
            reward_scale=reward_scale,
            temperature=temperature,
            soft_lr=soft_lr
        )
    
    def initialization(self):
        event = np.empty(self.e_dim)
        state, info = self.env.reset()
        
        for _ in range(0, self.init_steps):            
            action = np.random.rand(self.a_dim)*2 - 1
            next_state, reward, _, _, _ = self.env.step(self.scale_action(action))

            event[:self.s_dim] = state
            event[self.s_dim:self.sa_dim] = action
            event[self.sa_dim] = reward
            event[self.sa_dim+1:self.e_dim] = next_state          

            self.agent.memorize(event)
            state = np.copy(next_state)

    def scale_action(self, a):
        return (0.5 * (a + 1.0) * (self.max_action - self.min_action) + self.min_action)
    
    def interaction(self, state, learn=True, remember=True):   
        event = np.empty(self.e_dim)

        for _ in range(0, self.env_steps):
            cuda_state = torch.FloatTensor(state).unsqueeze(0).to(device)         
            action = self.agent.act(cuda_state, explore=learn)
            scaled_action = self.scale_action(action.detach().cpu().numpy().flatten())
            obs, reward, terminated, truncated, _ = self.env.step(scaled_action)
            done = terminated or truncated

            event[:self.s_dim] = state
            event[self.s_dim:self.sa_dim] = action
            event[self.sa_dim] = reward
            event[self.sa_dim+1:self.e_dim] = obs

            if remember:
                self.agent.memorize(event)
            
            state = np.copy(obs)
        
        if learn:
            for grad_step in range(0, self.grad_steps):
                self.agent.learn()
        
        return done, event, obs
    
    def train_agent(self, total_steps, initialization=True):
        if initialization: # TODO
            self.initialization()
        
        min_reward = 1e10
        max_reward = -1e10
        mean_reward = 0.0   
        min_mean_reward = 1e10
        max_mean_reward = -1e10   

        results = []
        steps_performed = 0
        epsd = 0
        
        while steps_performed < total_steps:
            epsd_min_reward = 1e10
            epsd_max_reward = -1e10                
            epsd_total_reward = 0.0

            obs, _ = self.env.reset()
            
            for step in count():
                if len(self.agent.memory.data) < self.batch_size:
                    done, event, obs = self.interaction(obs, learn=False)
                else:
                    done, event, obs = self.interaction(obs)
                r = event[self.sa_dim]

                min_reward = np.min([r, min_reward])
                max_reward = np.max([r, max_reward])
                epsd_min_reward = np.min([r, epsd_min_reward])                        
                epsd_max_reward = np.max([r, epsd_max_reward])                        
                epsd_total_reward += r  
                if done:
                    print(f"Terminated after {step} steps")
                    steps_performed += step + 1
                    break
            
            # if epsd_mean_reward > max_mean_reward:
            #     pickle.dump(self,open(self.type+'.p','wb'))
            
            epsd_mean_reward = epsd_total_reward / step
            results.append({"epsd_total_reward": epsd_total_reward, "final_step": step})

            min_mean_reward = np.min([epsd_mean_reward, min_mean_reward])
            max_mean_reward = np.max([epsd_mean_reward, max_mean_reward])
            mean_reward += (epsd_mean_reward - mean_reward)/(epsd+1)
            # print(f"Finished epsd {epsd+1}, epsd.min(r) = {epsd_min_reward:.4f}, epsd.max(r) = {epsd_max_reward:.4f}, min.(r) = {min_reward:.4f}, max.(r) = {max_reward:.4f}, min.(av.r) = {min_mean_reward:.4f}, max.(av.r) = {max_mean_reward:.4f}, epsd.av.r = {epsd_mean_reward:.4f}, total av.r = {mean_reward:.4f}\r") TODO
            print(f"Finished epsd {epsd+1}, step {steps_performed}, epsd_total_r: {epsd_total_reward:.4f}")
            epsd += 1
            time.sleep(0.0001)
        print("")     
        return results