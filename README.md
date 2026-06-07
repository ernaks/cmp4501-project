# CMP4501 – Autonomous Driving with Highway-Env

| | |
|---|---|
| **Student** | [Muhammed Eren Aksu 2203549 ] |
| **Course** | CMP4501 – Introduction to AI & Expert Systems |
| **Track** | Option A – Autonomous Driving with Highway-Env |
| **Algorithm** | Deep Q-Network (DQN) |

---

## Evolution: Untrained → Half-Trained → Fully Trained

| Untrained Agent | Half-Trained Agent | Fully Trained Agent |
|:---------:|:------------:|:-------------:|
| ![Untrained](videos/untrained.gif) | ![Half](videos/half_trained.gif) | ![Final](videos/final.gif) |

> The untrained agent moves randomly and crashes almost immediately. The half-trained agent survives longer but still makes poor lane decisions. The fully trained agent drives smoothly, avoids collisions, and maintains high speed.

---

## 1. States and Actions

### State Space

At each timestep, the agent observes a **5 × 5 matrix** representing the ego vehicle and the 4 nearest surrounding vehicles. Each row contains:

| Feature | Description |
|---------|-------------|
| `x` | Longitudinal position (normalized) |
| `y` | Lateral position (normalized) |
| `vx` | Longitudinal velocity (normalized) |
| `vy` | Lateral velocity (normalized) |
| `cos_h` | Cosine of heading angle |

This gives a flat observation vector of shape **(5, 5)** — fully numerical, no image processing required.

### Action Space

The agent selects from **5 discrete meta-actions**:

| Action | Description |
|--------|-------------|
| 0 – LANE_LEFT | Move one lane to the left |
| 1 – IDLE | Maintain current speed and lane |
| 2 – LANE_RIGHT | Move one lane to the right |
| 3 – FASTER | Increase speed |
| 4 – SLOWER | Decrease speed |

---

## 2. Methodology

### a. Custom Reward Function

The reward function balances speed and safety simultaneously:

```math
r(s, a) = w_1 \cdot r_{\text{speed}} + w_2 \cdot r_{\text{collision}} + w_3 \cdot r_{\text{right\_lane}}
```

Where each term is defined as:

```math
r_{\text{speed}} = \frac{v - v_{\min}}{v_{\max} - v_{\min}}, \quad v \in [20, 30] \text{ m/s}
```

```math
r_{\text{collision}} = \begin{cases} -2.0 & \text{if collision detected} \\ 0 & \text{otherwise} \end{cases}
```

```math
r_{\text{right\_lane}} = 0.1 \times \mathbb{1}[\text{agent is in rightmost lane}]
```

| Weight | Value | Justification |
|--------|-------|---------------|
| $w_1$ – speed | 1.0 | Primary objective: drive fast |
| $w_2$ – collision | −2.0 | Strong penalty to discourage crashes |
| $w_3$ – right lane | 0.1 | Small bonus to encourage safe lane discipline |

**Why this reward function?** The collision penalty is twice the maximum speed reward. This ensures the agent never finds it beneficial to speed through traffic recklessly. The right-lane bonus is intentionally small so it does not override the speed objective — it merely acts as a tiebreaker.

### b. Algorithm – Deep Q-Network (DQN)

**Why DQN?** The action space is discrete (5 actions) and the state space is a small numerical vector (5×5). DQN is well-suited for discrete action spaces and converges efficiently on low-dimensional inputs without requiring a CNN.

**Neural Network Architecture:**
- Input layer: 25 neurons (flattened 5×5 observation)
- Hidden layers: 2 fully connected layers (64 neurons each)
- Activation: ReLU
- Output layer: 5 neurons (one Q-value per action)

**Key Hyperparameters:**

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Learning rate | 5e-4 | Small enough for stable convergence |
| Batch size | 32 | Memory efficient for CPU training |
| Buffer size | 15,000 | Sufficient replay diversity |
| Gamma (γ) | 0.8 | Moderate discounting — short-horizon task |
| Exploration fraction | 0.7 | Long exploration phase for diverse experience |
| Final epsilon (ε) | 0.1 | Retains slight randomness to avoid overfitting |
| Target update interval | 50 | Stabilizes Q-value targets |
| Total timesteps | 100,000 | Sufficient for convergence on CPU |

---

## 3. Training Analysis

### Reward Graph

![Reward Curve](assets/reward_plot.png)

### Commentary

The training curve shows a **clear upward trend** across approximately 5,800 episodes.

- **Episodes 0–1000:** The agent explores aggressively (ε ≈ 0.7–1.0). Rewards are low and highly variable because most actions are random. The moving average stays around 8–10.
- **Episodes 1000–3000:** As exploration decreases, the agent begins exploiting learned patterns. The moving average climbs steadily from ~10 to ~18, showing that the agent learned to avoid immediate collisions.
- **Episodes 3000–5800:** Learning stabilizes. The moving average reaches ~22–25. The agent consistently avoids crashes and maintains higher speeds. Performance plateaus near the end, suggesting the policy is close to convergence for 100k timesteps.

The learning was **relatively stable** with no major collapses, which reflects the benefit of experience replay and target network stabilization in DQN.

---

## 4. Challenges and Failures

### Challenge 1: Observation Shape Mismatch
When the project was first run, training crashed immediately with a `ValueError: could not broadcast input array from shape (5,6) into shape (5,5)`. The config initially specified 6 features, but the environment's default observation buffer expected 5. This was resolved by removing the `sin_h` feature from the observation config.

### Challenge 2: No Visual Difference Between Training Stages
Early GIF recordings showed all three agents behaving similarly — none crashed visibly. The root cause was that the untrained agent used `model.predict()` with zero weights, which produced calm but uninformed behavior rather than truly random actions. This was fixed by switching the untrained agent to `env.action_space.sample()` for fully random behavior, and adjusting traffic density per stage to make differences more visually apparent.

### Challenge 3: Slow Training on CPU
Training 100,000 timesteps on CPU took approximately 40–50 minutes. To keep this feasible, DQN was chosen over PPO (which requires more environment interactions), and the replay buffer was kept small (15,000) to reduce memory overhead.

---

## 5. Project Structure

```
cmp4501-project/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py        # Environment & training configuration
│   ├── model.py         # DQN model definition
│   ├── train.py         # Training script with checkpoints
│   ├── evaluate.py      # Evaluation script
│   ├── record_video.py  # GIF recording script
│   └── utils.py         # Reward logger & plotting
├── models/
│   ├── untrained.zip
│   ├── half_trained.zip
│   └── final.zip
├── assets/
│   └── reward_plot.png
└── videos/
    ├── untrained.gif
    ├── half_trained.gif
    └── final.gif
```

---

## 6. Setup & Run

```bash
# Create virtual environment
py -3.11 -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the agent
python -m src.train

# Record evolution videos
python -m src.record_video

# Evaluate final model
python -m src.evaluate
```