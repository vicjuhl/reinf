import numpy as np
from time import sleep

def run_pi(γ, π, p_est, r_est, rand_level, i):
    """Policy iteration, using existing π as starting point."""
    n_s = r_est.shape[1]
    p_π = p_est[π, np.arange(n_s)]
    r_π = r_est[π, np.arange(n_s)].reshape(-1, 1)

    while True:
        π_old = π.copy()
        v = np.linalg.solve(np.eye(n_s) - γ * p_π, r_π)
        for s in range(n_s):
            noise = rand_level * np.random.uniform(-1, 1, (2, 1)) / (i + 1)
            q_s = r_est[:, s].reshape(-1, 1) + γ * p_est[:, s, :] @ v + noise
            π[s] = np.argmax(q_s)
        if np.all(π == π_old):
            return π

def construct_mdp(h_t, h_r, m, r_max):
    n_a = h_t.shape[0]
    n_s = h_t.shape[1]
    p_est = np.zeros_like(h_t)
    r_est = np.zeros_like(h_r)

    explored = h_t.sum(axis=2) >= m
    # Index both action and state dimensions using explored mask
    for a in range(n_a):
        for s in range(n_s):
            if explored[a, s]:
                # Compute empirical fractions
                p_est[a, s, :] = h_t[a, s, :] / h_t[a, s, :].sum()
                r_est[a, s] = h_r[a, s] / h_t[a, s, :].sum()
            else:
                # Use defaults according to the R-max algorithm
                p_est[a, s, :] = np.eye(n_s)[s] # One hot for current state
                r_est[a, s] = r_max
    return p_est, r_est

def simulate(
    k: int,
    period: int,
    river_length: int,
    γ: float,
    m: int,
    r_max: float,
    rand_pol: float,
    rand_act: float,
):
    n_s = river_length + 1 # number of states

    # Counting realized transitions for each
    # 2 actions x (from) n_states x (to) n_states
    h_t = np.zeros((2, n_s, n_s), dtype=float) # float for ease of division later
    
    # Deterministic transitions (Δs) based on state and realized move
    transitions = np.array([
        [0, *[-1 for _ in range(river_length)]], # left
        [0 for _ in range(n_s)], # stay
        [*[1 for _ in range(river_length)], 0]  # right
    ], dtype=int)

    # True probability matrix for action "right" (==1)
    p_right = np.array([
        [0, 0.4, 0.6],
        *[[0.05, 0.6, 0.35] for _ in range(river_length - 1)],
        [0.4, 0, 0.6]
    ], dtype=float)

    # Rewards
    # Total rewards history (r_total[i, t] is the total reward gathered before step t after policy iteration i)
    r_total = np.ones((k, period + 1), dtype=float) * -1
    r_total[:, 0] = 0
    # Reward history for each (a,s)
    h_r = np.zeros((2, n_s))
    # True rewards based on state and realized move
    rewards = np.array([
        [5/1000, *[0 for _ in range(river_length)]], # left
        [0 for _ in range(n_s)], # stay
        [*[0 for _ in range(river_length)], 1] # right
    ], dtype=float)

    π = np.random.randint(0, 2, n_s) # n_s random integers in {0, 1}
    π_star_learned = -1 # store later when optimal policy was learned

    best_rolling_avg = 0 # of last five total rewards
    best_rolling_avg_idx = 0 # corresponding index
    
    # Run simulation, limited to k iterations
    for i in range(k):
        p_est, r_est = construct_mdp(h_t, h_r, m, r_max)
        π = run_pi(γ, π, p_est, r_est, rand_pol, i)
        if π_star_learned == -1 and np.min(π) == 1: # optimal policy learned now
            π_star_learned = i
            print("π_star learned at iteration", i)
        
        s = int(0)
        for t in range(period):
            # Find action deterministically
            a = π[s]
            # ... and possibly flip stochastically
            p_flip = rand_act / (i+2)**((35-river_length) / 15)
            flip = np.random.choice(2, 1, p=[1 - p_flip, p_flip])
            if flip:
                a = int(not a)
            if a: # right (π == 1)
                # 3 indicates arange(3) where 0 = left, 1 = stay, 2 = right
                move = np.random.choice(3, 1, p=p_right[s])
            else: # left (π == 0)
                move = 0 # left

            # Updates
            # Reward
            r = rewards[move, s]
            h_r[a, s] += r
            r_total[i, t+1] = r_total[i, t] + r
            # State transition
            s_old = s
            s += transitions[move, s].item() # Δs based on move
            h_t[a, s_old, s] += 1

        if i >= 6:
            rolling_avg = np.mean(r_total[(i-5):i,-1])
            if rolling_avg > best_rolling_avg:
                best_rolling_avg = rolling_avg
                best_rolling_avg_idx = i
            worst_of_last_few = np.min(r_total[(i-5):i,-1])
            if (
                worst_of_last_few > period * 0.005 * 5 # Above threshold
                and best_rolling_avg_idx + 5 <= i # No new record (of recent avg) for a few iterations ("convergence")
            ):
                print(f"Ending simulation after {i} iterations")
                break
    
    return r_total, π_star_learned, i