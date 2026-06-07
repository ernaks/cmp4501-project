from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import highway_env
from src.config import ENV_CONFIG, TRAIN_CONFIG


def make_env() -> gym.Env:
    """Create and configure the highway environment."""
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.config.update(ENV_CONFIG)
    return env


def create_model(env: gym.Env) -> DQN:
    """Create a DQN model with the given environment."""
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,
        batch_size=32,
        buffer_size=15_000,
        learning_starts=200,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        exploration_fraction=0.7,
        exploration_final_eps=0.1,
        verbose=1,
        tensorboard_log=TRAIN_CONFIG["log_path"],
    )
    return model


def load_model(path: str, env: gym.Env) -> DQN:
    """Load a saved DQN model."""
    return DQN.load(path, env=env)