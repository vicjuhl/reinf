import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path

MODELS = {
    "sis": "e-sarsa_importance_sampling",
    "snis": "e-sarsa",
    "q": "q"
}
NICE_MODEL_NAMES = {
    "sis": "Exp. SARSA with Imp. Sampling",
    "snis": "Exp. SARSA without Imp. Sampling",
    "q": "Q-learning"
}
GAMES = [
    "beamrider",
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
    # Check if preprocessed results exist
    results_dir = Path("final_results")
    tr_path = results_dir / "training_results_prep.csv" 
    eval_path = results_dir / "eval_agg_results.csv"
    
    if tr_path.exists() and eval_path.exists():
        tr_res = pd.read_csv(tr_path)
        eval_agg = pd.read_csv(eval_path)
        return tr_res, eval_agg
    tr_res, eval_res = load_results()
    
    # Calculate aggregated statistics for each game, model, exp_id, epoch (aggregating over five evaluations)
    eval_agg = eval_res.groupby(['game', 'model', 'exp_id', 'epoch']).agg({
        'reward': ['mean']
    }).reset_index()
    # Flatten column names
    eval_agg.columns = ['game', 'model', 'exp_id', 'epoch', 'avg_reward']

    eval_agg = eval_agg.groupby(['game', 'model', 'epoch']).agg({
        'avg_reward': ['min', 'max', 'mean']
    }).reset_index()
    # Flatten column names
    eval_agg.columns = ['game', 'model', 'epoch', 'min_reward', 'max_reward', 'avg_reward']
    
    eval_path.parent.mkdir(exist_ok=True)
    eval_agg.to_csv(eval_path)
    tr_res.to_csv(tr_path)
    return tr_res, eval_agg

def plot_results():
    tr_res, eval_agg = prepare_results_for_plots()
    fig_path = Path("plots")
    fig_path.mkdir(exist_ok=True)

    for (game, width) in zip(GAMES, [2, 2, 1]):
        plt.figure(figsize=(10,6))
        for model, col in zip(MODELS.keys(), ["red", "blue", "green"]):
            model_res = eval_agg[(eval_agg['game'] == game) & (eval_agg['model'] == model)]
            
            # Plot one line per aggregation
            # for agg in ['min', 'max']:
            #     plt.plot(model_res['epoch'], model_res[f"{agg}_reward"], color=col, linewidth=width/3, linestyle="--")
            plt.plot(model_res['epoch'], model_res['avg_reward'], color=col, linewidth=width, label=NICE_MODEL_NAMES[model])
        
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Average Evaluation Reward', fontsize=16)
        plt.title(f'{str.upper(game)}', fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.savefig(fig_path / f"{game}.png")

plot_results()