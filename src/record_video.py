import os
import random
import gymnasium as gym
import highway_env
import imageio
from stable_baselines3 import DQN


UNTRAINED_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["x", "y", "vx", "vy", "cos_h"],
        "normalize": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 3,
    "vehicles_count": 25,
    "duration": 60,
    "collision_reward": -2.0,
    "high_speed_reward": 1.0,
    "lane_change_reward": 0.0,
    "right_lane_reward": 0.1,
}

HALF_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["x", "y", "vx", "vy", "cos_h"],
        "normalize": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 20,
    "duration": 60,
    "collision_reward": -2.0,
    "high_speed_reward": 1.0,
    "lane_change_reward": 0.0,
    "right_lane_reward": 0.1,
}

FINAL_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["x", "y", "vx", "vy", "cos_h"],
        "normalize": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 10,
    "duration": 60,
    "collision_reward": -2.0,
    "high_speed_reward": 1.0,
    "lane_change_reward": 0.0,
    "right_lane_reward": 0.1,
}


def record_episode(model_path, config, output_path, max_steps=400):
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.config.update(config)

    frames = []
    obs, _ = env.reset()
    steps = 0

    if model_path is None:
        while steps < max_steps:
            frame = env.render()
            frames.append(frame)
            action = random.choices([0, 1, 2, 3, 4], weights=[3, 0, 3, 4, 0])[0]
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                for _ in range(20):
                    frames.append(env.render())
                break
            steps += 1
    else:
        model = DQN.load(model_path, env=env)
        while steps < max_steps:
            frame = env.render()
            frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                for _ in range(20):
                    frames.append(env.render())
                break
            steps += 1

    env.close()
    imageio.mimsave(output_path, frames, fps=10)
    print(f"Saved: {output_path} ({len(frames)} frames)")


def record_all():
    os.makedirs("videos", exist_ok=True)

    print("Recording untrained agent (random)...")
    record_episode(None, UNTRAINED_CONFIG, "videos/untrained.gif")

    print("Recording half-trained agent...")
    record_episode("models/half_trained", HALF_CONFIG, "videos/half_trained.gif")

    print("Recording fully trained agent...")
    record_episode("models/final", FINAL_CONFIG, "videos/final.gif")

    print("All videos recorded!")


if __name__ == "__main__":
    record_all()
