import pathlib as pl

SCRIPT_DIR = pl.Path(__file__).parent.absolute()
MODELS_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
VIDEOS_DIR = SCRIPT_DIR / "videos"
PLOTS_DIR = SCRIPT_DIR / "plots"

REWARD_SCALE = {
    "Pendulum-v1": 5,
    "Hopper-v4": 100,
    "HalfCheetah-v4": 3,
    "Swimmer-v4": 5
}

EPISODE_LENGTH = {
    "Pendulum-v1": 200,
    "Hopper-v4": 1000,
}

TOTAL_STEPS = {
    "Pendulum-v1": int(2e4),
    "Hopper-v4": int(1e6),
}