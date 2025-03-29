import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path

MODELS = {
    "sis": "e-sarsa_importance_sampling",
    "snis": "e-sarsa",
    "q": "q"
}
GAMES = [
    # "beamrider",
    "boxing",
    "breakout"
]

def load_inner_results(game: str, model: str, exp_id: int) -> tuple[dict, pd.DataFrame]:
    """
    Load results.json and eval_rewards.csv for a specific game and model combination
    
    Args:
        game (str): Game name (beamrider/boxing/breakout)
        model (str): Model type (double_e-sarsa/double_e-sarsa_importance_sampling/double_q)
    
    Returns:
        tuple: (results_dict, eval_rewards_df)
    """
    # Construct paths
    game_path = Path(f"log_{game}")
    model_path = game_path / f"double_{model}"
    exp_path = model_path / f"exp_{exp_id}"
    results_path = exp_path / "results.json"
    eval_path = exp_path / "eval_rewards.csv"
    
    # Load JSON results
    with results_path.open('r') as f:
        results_dict = json.load(f)
    
    # Load evaluation rewards
    eval_rewards_df = pd.read_csv(eval_path)
    
    return results_dict, eval_rewards_df

def load_results():
    tr_results = []
    eval_results = []
    for game in GAMES:
        for model_short, model_long in MODELS.items():
            for exp_id in (1, 2, 3):
                tr_res, ev_res = load_inner_results(game, model_long, exp_id)
                for t in range(len(tr_res["rewards"])):
                    tr_results.append((
                        game, model_short, exp_id, t,
                        tr_res["rewards"][t], tr_res["losses"][t],
                        tr_res["avg_rewards"][t], tr_res["avg_losses"][t]
                    ))
                    eval_results.append((game, model_short, exp_id, ev_res))

    df_tr_res = pd.DataFrame(tr_results, columns=[
        "game", "model", "exp_id", "t",
        "rewards", "losses", "avg_rewards", "avg_losses"
    ])
    df_eval_res = pd.concat(
        [
            df.assign(game=game, model=model, exp_id=exp_id)
            for game, model, exp_id, df in eval_results
        ], ignore_index=True
    )
    return df_tr_res, df_eval_res

def prepare_results_for_plots():
    tr_res, eval_res = load_results()
    
    # Calculate aggregated statistics grouped by game and model
    eval_agg = eval_res.groupby(['game', 'model', 'epoch']).agg({
        'reward': ['min', 'max', 'mean']
    }).reset_index()
    # Flatten column names
    eval_agg.columns = ['game', 'model', 'epoch', 'min_reward', 'max_reward', 'avg_reward']
    
    print(tr_res, eval_agg)
    return tr_res, eval_agg

def plot_results():
    tr_res, eval_agg = prepare_results_for_plots()
    # Plot evaluation results for Boxing Q-learning experiments
    plt.figure(figsize=(10,6))
    
    # Filter for boxing and Q-learning model
    boxing_q = eval_agg[(eval_agg['game'] == 'boxing') & (eval_agg['model'] == 'q')]
    
    # Plot one line per experiment
    for agg in ['min', 'max']:
        plt.plot(boxing_q['epoch'], boxing_q[f"{agg}_reward"], color="red", linewidth=0.5)
    plt.plot(boxing_q['epoch'], boxing_q['avg_reward'], color="red", linewidth=3)
    
    plt.xlabel('Epochs')
    plt.ylabel('Evaluation Reward')
    plt.title('Boxing Q-Learning Evaluation Results')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_results()