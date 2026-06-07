import numpy as np
import gymnasium as gym
import highway_env
from src.model import make_env, load_model, create_model
from src.config import EVAL_CONFIG, TRAIN_CONFIG


def evaluate_model(model_path: str | None, n_episodes: int = 5) -> list[float]:
    """Evaluate a model and return episode rewards."""
    env = make_env()

    if model_path is None:
        model = create_model(env)
    else:
        model = load_model(model_path, env)

    rewards: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward: float = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        rewards.append(total_reward)
        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")

    env.close()
    return rewards


if __name__ == "__main__":
    evaluate_model(TRAIN_CONFIG["checkpoint_final"])