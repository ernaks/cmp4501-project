import os
import gymnasium as gym
import highway_env
import numpy as np
import imageio
from src.model import make_env, load_model, create_model
from src.config import TRAIN_CONFIG


def record_episode(model_path: str | None, output_path: str, max_steps: int = 200) -> None:
    """Record a single episode and save as GIF."""
    env = gym.make("highway-v0", render_mode="rgb_array")
    from src.config import ENV_CONFIG
    env.unwrapped.config.update(ENV_CONFIG)

    if model_path is None:
        model = create_model(env)
    else:
        model = load_model(model_path, env)

    frames: list = []
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        frame = env.render()
        frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1

    env.close()
    imageio.mimsave(output_path, frames, fps=15)
    print(f"Saved: {output_path}")


def record_all() -> None:
    """Record untrained, half-trained, and fully trained episodes."""
    os.makedirs("videos", exist_ok=True)

    print("Recording untrained agent...")
    record_episode(None, "videos/untrained.gif")

    print("Recording half-trained agent...")
    record_episode(TRAIN_CONFIG["checkpoint_half"], "videos/half_trained.gif")

    print("Recording fully trained agent...")
    record_episode(TRAIN_CONFIG["checkpoint_final"], "videos/final.gif")

    print("All videos recorded!")


if __name__ == "__main__":
    record_all()