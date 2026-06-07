import os
from src.model import make_env, create_model
from src.config import TRAIN_CONFIG
from src.utils import RewardLoggerCallback, plot_rewards


def train() -> None:
    """Train the DQN agent and save checkpoints."""
    os.makedirs(TRAIN_CONFIG["model_save_path"], exist_ok=True)
    os.makedirs(TRAIN_CONFIG["log_path"], exist_ok=True)

    env = make_env()
    model = create_model(env)
    callback = RewardLoggerCallback()

    total: int = TRAIN_CONFIG["total_timesteps"]
    half: int = total // 2

    # Save untrained checkpoint
    model.save(TRAIN_CONFIG["checkpoint_untrained"])
    print("Saved: untrained checkpoint")

    # Train to halfway point
    model.learn(total_timesteps=half, reset_num_timesteps=True, callback=callback)
    model.save(TRAIN_CONFIG["checkpoint_half"])
    print("Saved: half-trained checkpoint")

    # Train to completion
    model.learn(total_timesteps=half, reset_num_timesteps=False, callback=callback)
    model.save(TRAIN_CONFIG["checkpoint_final"])
    print("Saved: final checkpoint")

    # Plot rewards
    plot_rewards(callback.episode_rewards)

    env.close()
    print("Training complete!")


if __name__ == "__main__":
    train()