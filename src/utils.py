import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    """Callback to log episode rewards during training."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self._current_reward: float = 0.0

    def _on_step(self) -> bool:
        self._current_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True


def plot_rewards(rewards: list[float], save_path: str = "assets/reward_plot.png") -> None:
    """Plot and save the reward curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.4, color="steelblue", label="Episode Reward")

    window = min(20, len(rewards))
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), moving_avg,
                color="darkorange", linewidth=2, label=f"{window}-ep Moving Avg")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Reward Curve – Highway-Env DQN")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Reward plot saved to {save_path}")