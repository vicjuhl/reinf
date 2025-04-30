import pathlib as pl

SCRIPT_DIR = pl.Path(__file__).parent.absolute()
MODELS_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
VIDEOS_DIR = SCRIPT_DIR / "videos"
REWARD_SCALE = {
    "Pendulum-v1": 5,
    "Hopper-v4": 10,
    "HalfCheetah-v4": 3,
    "Swimmer-v4": 5
}