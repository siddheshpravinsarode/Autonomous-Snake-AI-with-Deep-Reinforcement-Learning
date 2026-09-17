# Autonomous Snake AI with Deep Reinforcement Learning (Double DQN)

An autonomous Snake agent trained from scratch using **Double Deep Q-Networks (Double DQN)** in PyTorch and Pygame.

---

## ⚡ Quick Start (Play Immediately)

Anyone cloning this repository can run the pre-trained AI in two commands:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the AI Snake game (loads pre-trained models/dqn_snake.pth automatically)
python main.py
```

---

## Why We Switched to Deep Reinforcement Learning

In earlier iterations of this project, we experimented with a biologically-inspired fruit fly connectome using Hebbian dopamine plasticity. While fascinating in theory, simple Hebbian synaptic updates fail at games like Snake because:
- They only reinforce immediate co-active firing (no long-term planning).
- They have no concept of future discounted rewards (Bellman equation).
- The snake would continually trap itself, run into walls, or spin in circles.

We replaced the heuristic approach with **Double Deep Q-Learning (Double DQN)** coupled with an **ego-centric (head-relative) state representation**. Within just **150 headless training episodes (~2 minutes)**, the agent learned collision avoidance, path finding, and achieved scores of **44+ fruits**.

---

## Project Structure

We modularized the previous monolithic script into clean, dedicated components:

```
flyslave/
├── models/
│   └── dqn_snake.pth       # Trained PyTorch model weights
├── src/
│   ├── constants.py        # Window dimensions, colors, grid sizes, directions
│   ├── game.py             # Snake physics, grid state, collisions, reward shaping
│   ├── hud.py              # Real-time HUD (Q-value bars, telemetry, radar badges)
│   ├── renderer.py         # Pygame arena drawing, glowing food, particle effects
│   ├── rl_agent.py         # PyTorch DQN neural net, ReplayBuffer, Q-learning updates
│   └── sensors.py          # Ego-centric hazard detection & proximity state encoder
├── main.py                 # Interactive GUI game loop with real-time controls
├── train_headless.py       # Super-fast headless training script (300+ steps/sec)
├── fly_snake_game.py       # Wrapper entrypoint calling main.py
└── README.md               # Documentation
```

---

## Function-by-Function Breakdown of the RL System

Here is an exact walkthrough of how every Reinforcement Learning function works under the hood:

### 1. Neural Network & Experience Replay ([`src/rl_agent.py`](src/rl_agent.py))

#### `QNetwork`
- **What it is**: A standard multi-layer perceptron (MLP) built with PyTorch `nn.Module`.
- **Architecture**: `16 inputs` $\to$ `Linear(16, 128)` $\to$ `ReLU()` $\to$ `Linear(128, 128)` $\to$ `ReLU()` $\to$ `Linear(128, 3 actions)`.
- **What it outputs**: Three continuous Q-values: $Q(s, \text{Turn Left})$, $Q(s, \text{Straight})$, $Q(s, \text{Turn Right})$. The action with the highest Q-value is chosen as the best move.

#### `ReplayBuffer.push(state, action, reward, next_state, done)`
- **What it does**: Saves every step the snake takes into a circular queue (`deque` with a capacity of 50,000 transitions).
- **Why it matters**: If an RL model only trains on consecutive frames, the data is heavily correlated (the snake just moved 1 pixel), leading to overfitting or catastrophic forgetting. Replay memory stores experiences so the network can sample them randomly later.

#### `ReplayBuffer.sample(batch_size=64)`
- **What it does**: Pulls 64 random transitions from memory and converts them into PyTorch FloatTensors on CPU or GPU.
- **Why it matters**: Breaks temporal correlation and provides independently distributed data for gradient descent.

#### `DQNAgent.__init__(...)`
- **What it sets up**:
  - `self.policy_net`: The active network that makes decisions and gets updated on every step.
  - `self.target_net`: A cloned copy used exclusively to calculate target values for Bellman updates.
  - `self.optimizer`: Adam optimizer with learning rate $\alpha = 10^{-3}$.
  - `self.loss_fn`: Huber Smooth-L1 Loss (less sensitive to outliers than MSE).
  - Exploration rate ($\epsilon = 1.0 \to 0.02$ decay).

#### `DQNAgent.select_action(state, evaluate=False)`
- **What it does**: Implements the **$\epsilon$-greedy exploration strategy**:
  - With probability $\epsilon$, pick a random action (explores new paths).
  - With probability $1 - \epsilon$, pass the state through `policy_net` and pick `argmax(Q)` (exploits learned knowledge).
  - When `evaluate=True` (during normal gameplay viewing), randomness is turned off to execute strictly optimal moves.

#### `DQNAgent.train_step(batch_size=64)`
- **What it does**: This is the heart of the learning algorithm:
  1. Samples 64 random transitions `(s, a, r, s', done)` from replay memory.
  2. Computes the predicted Q-value for the action that was taken: $Q_{policy}(s, a)$.
  3. Uses **Double DQN** logic to determine the target value:
     $$a^* = \arg\max_{a'} Q_{policy}(s', a')$$
     $$y = r + (1 - done) \cdot \gamma \cdot Q_{target}(s', a^*)$$
  4. Calculates Smooth-L1 Loss between prediction and target.
  5. Performs `loss.backward()`, clips gradients to `max_norm=5.0` (to avoid exploding gradients), and calls `optimizer.step()`.
  6. Decays $\epsilon$ by $0.995$ and syncs the target network every 150 steps.

#### `DQNAgent.update_target_network()`
- **What it does**: Copies `policy_net.state_dict()` into `target_net`.
- **Why it matters**: If you update the target values with the same network you are training, the targets constantly move, causing unstable oscillations. Freezing the target network and updating it periodically keeps training rock solid.

#### `DQNAgent.save(filepath)` & `load(filepath)`
- **What it does**: Saves/loads model weights, optimizer state, exploration epsilon, and total step count to `.pth` files.

---

### 2. Sensory State Perception ([`src/sensors.py`](src/sensors.py))

#### `extract_rl_state(game)`
- **What it does**: Transforms the full 28x28 grid into a compact **16-dimensional ego-centric observation vector**:
  - `[0..2] Danger (Straight, Left, Right)`: Look-ahead radar that checks if moving into the adjacent cell hits a wall or body segment.
  - `[3..6] Direction (Up, Right, Down, Left)`: One-hot vector of current heading.
  - `[7..10] Food Direction (Ahead, Left, Right, Behind)`: Projection of the food vector onto the snake's forward and lateral axes.
  - `[11..13] Food Proximity`: Inverse-distance values at the head, left sensor offset, and right sensor offset.
  - `[14] Progress Trend`: Flag indicating whether the previous step moved closer to food.
  - `[15] Normalized Distance`: Straight-line Euclidean distance to food divided by arena diagonal.

#### Why Ego-Centric State?
If we gave the network absolute coordinates $(x_{head}, y_{head}, x_{food}, y_{food})$, the agent would have to re-learn how to turn towards food separately for every quadrant of the board. With **ego-centric coordinates**, "food is to my left" means the exact same thing whether the snake is heading North, East, South, or West. This **rotation invariance** reduces training time from days to minutes.

---

### 3. Environment & Reward Shaping ([`src/game.py`](src/game.py))

#### `SnakeGame.step(action)`
- **Actions**:
  - `0`: Turn Left (-90°)
  - `1`: Go Straight (0°)
  - `2`: Turn Right (+90°)
- **Reward System**:
  - **$+10.0$**: Eating food (primary objective).
  - **$-10.0$**: Colliding with wall or self (death).
  - **$+0.15$ to $+0.30$**: Moving closer to food / climbing the scent plume (encourages direct navigation).
  - **$-0.20$**: Moving away from food (discourages wandering).
  - **$-8.0$**: Starvation penalty if steps without food exceed $(120 + 10 \times \text{length})$ (prevents infinite harmless loops).

---

## Running the Application

### 1. Interactive Visual GUI
Watch the agent play live with real-time HUD telemetry:
```bash
python main.py
```
*(Alternatively, `python fly_snake_game.py` works as an alias).*

#### Key Controls:
| Key | Action |
| :--- | :--- |
| **`[M]`** | Toggle between **AI Auto-Play** and **Manual Keyboard Control** |
| **`[F]`** | Toggle **Fast Training Mode** (140 FPS) |
| **`[O]`** | Toggle Target Proximity Diffusion Heatmap |
| **`[S]`** | Manually save network weights to `models/dqn_snake.pth` |
| **`[SPACE]`** | Restart / reset episode |
| **`[UP / DOWN]`** | Increase / decrease simulation speed |
| **`[LEFT / RIGHT]`** | Steer snake (when in Manual mode) |

---

### 2. High-Speed Headless Training
To train or improve the model without Pygame graphics overhead:
```bash
python train_headless.py --episodes 300
```
- Runs at **300+ steps per second**.
- Outputs average score, loss, and epsilon every 25 episodes.
- Automatically saves new weights to `models/dqn_snake.pth`.

---

## Results & Benchmarks

| Metric | Previous Hebbian Model | Deep Q-Network (Our Model) |
| :--- | :--- | :--- |
| **Average Score** | ~0 - 1 fruits | **12 - 16 fruits** |
| **High Score** | 3 fruits | **44+ fruits** |
| **Convergence Time** | Did not converge | **134 seconds (150 episodes)** |
| **Movement Behavior** | Random spinning, wall crashes | **Direct vectoring, self-body avoidance** |
