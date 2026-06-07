# Configuration file for the Highway-Env RL project

ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["x", "y", "vx", "vy", "cos_h"],
        "normalize": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 4,
    "vehicles_count": 20,
    "duration": 40,
    "reward_speed_range": [20, 30],
    "collision_reward": -2.0,
    "high_speed_reward": 1.0,
    "lane_change_reward": 0.0,
    "right_lane_reward": 0.1,
}

TRAIN_CONFIG = {
    "total_timesteps": 100_000,
    "model_save_path": "models/",
    "log_path": "logs/",
    "checkpoint_untrained": "models/untrained",
    "checkpoint_half": "models/half_trained",
    "checkpoint_final": "models/final",
}

EVAL_CONFIG = {
    "n_episodes": 5,
    "video_length": 200,
}