# Autonomous Snake AI with Deep Reinforcement Learning

A Snake game that learns to play by itself using **Double Deep Q-Learning (Double DQN)** with PyTorch and Pygame.

The agent starts with no knowledge of how to play and learns by interacting with the game, receiving rewards for useful actions and penalties for bad ones.

## Quick Start 0

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then start the game:

```bash
python main.py
```

The game automatically loads the trained model from:

```text
models/dqn_snake.pth
```

---

## Why Deep Reinforcement Learning?

I originally experimented with a biologically-inspired approach using a **fruit fly connectome and Hebbian dopamine plasticity**.

It was an interesting idea, but it didn't work well for Snake. The main issue was that the approach mostly reinforced things happening at the current moment. It didn't have a good way to learn that an action can be bad several moves later.

The snake would often:

* Run into walls
* Trap itself
* Move in circles
* Wander without reaching the food

I switched to **Double DQN** to give the agent a way to estimate the future value of its actions.

I also changed the state representation to an **ego-centric representation**. Instead of giving the network only absolute coordinates, the snake gets information relative to its own direction, such as whether food is ahead or to the left and whether there is danger nearby.

With the current setup, the agent was able to reach **44+ fruits in a run after around 150 headless training episodes**.

---

## Project Structure

```text
flyslave/
│
├── models/
│   └── dqn_snake.pth
│
├── src/
│   ├── constants.py
│   ├── game.py
│   ├── hud.py
│   ├── renderer.py
│   ├── rl_agent.py
│   └── sensors.py
│
├── main.py
├── train_headless.py
├── fly_snake_game.py
└── README.md
```

### `src/constants.py`

Contains the game constants such as window size, grid size, colors, and movement directions.

### `src/game.py`

Contains the actual Snake game logic:

* Snake movement
* Food spawning
* Collision detection
* Rewards
* Episode termination

### `src/sensors.py`

Converts the game state into the 16 values that are given to the neural network.

### `src/rl_agent.py`

Contains the reinforcement learning code:

* DQN network
* Replay buffer
* Epsilon-greedy action selection
* Double DQN training
* Target network
* Model saving and loading

### `src/renderer.py`

Handles the Pygame rendering of the Snake game.

### `src/hud.py`

Displays information about the agent while it is playing, including Q-values and other telemetry.

### `train_headless.py`

Runs the environment without rendering the Pygame window so that training can run faster.

### `main.py`

Starts the interactive version of the game.

---

# How the RL Agent Works

## Neural Network

The agent uses a small fully connected neural network:

```text
16 inputs
    |
    v
Linear(16, 128)
    |
   ReLU
    |
    v
Linear(128, 128)
    |
   ReLU
    |
    v
Linear(128, 3)
```

The three outputs correspond to the three actions available to the snake:

```text
0 -> Turn Left
1 -> Go Straight
2 -> Turn Right
```

The network produces a Q-value for each action.

For example:

```text
Left      = 2.1
Straight  = 5.7
Right     = 3.4
```

The agent would choose **Straight** because it has the highest Q-value.

---

## Experience Replay

Every step of the game produces a transition:

```text
(state, action, reward, next_state, done)
```

These transitions are stored in a replay buffer with a capacity of **50,000 experiences**.

Instead of training only on the most recent move, the agent randomly samples a batch of experiences from the buffer.

The current training batch size is:

```text
64
```

This helps reduce the correlation between consecutive game states and gives the network a wider variety of situations to learn from.

---

## Epsilon-Greedy Exploration

At the beginning of training, the agent doesn't know which actions are good, so it needs to explore.

The agent uses epsilon-greedy action selection:

```text
With probability epsilon:
    choose a random action

Otherwise:
    choose the action with the highest Q-value
```

The exploration value starts at:

```text
epsilon = 1.0
```

and gradually decreases toward:

```text
epsilon = 0.02
```

This means the early training episodes contain a lot of random movement, while later episodes rely more on the behavior learned by the network.

---

# Double DQN

The main learning logic is inside:

```text
DQNAgent.train_step()
```

For each sampled transition, the policy network calculates:

```text
Q_policy(s, a)
```

For the next state, Double DQN uses the policy network to choose the next action:

```text
a* = argmax Q_policy(s', a)
```

The target network then evaluates that action:

```text
y = r + (1 - done) * gamma * Q_target(s', a*)
```

If the episode has ended, there is no future reward:

```text
y = r
```

The predicted Q-value is compared with the target using **Smooth L1 / Huber Loss**.

The network is then updated using backpropagation and the Adam optimizer.

Gradients are clipped to a maximum norm of `5.0` to prevent very large updates.

The target network is synchronized with the policy network every **150 training steps**.

---

# Policy Network and Target Network

The agent keeps two copies of the neural network.

```text
Policy Network
     |
     | makes decisions
     | gets updated during training
     v
  Actions


Target Network
     |
     | calculates training targets
     | updated periodically
     v
  Stable targets
```

The reason for having a separate target network is to make the training targets more stable.

If the same network were used to both generate and update the target values on every step, the values could keep moving while the network is trying to learn them.

---

# State Representation

The Snake board is a **28 x 28 grid**, but the neural network does not receive the whole grid.

Instead, the game converts the current situation into a **16-dimensional state vector**.

The values contain information about:

### 1. Danger

The snake checks the cells in three directions:

```text
Straight
Left
Right
```

These sensors detect whether the next move would hit a wall or the snake's body.

### 2. Current Direction

The current heading is represented using four values:

```text
Up
Right
Down
Left
```

### 3. Food Direction

The food is represented relative to the snake:

```text
Ahead
Left
Right
Behind
```

### 4. Food Proximity

The state contains additional proximity values for the food around the snake.

### 5. Progress

The agent keeps track of whether its previous movement brought it closer to the food.

### 6. Distance

The normalized Euclidean distance between the snake and the food is also included.

---

# Why Use an Ego-Centric State?

One option would be to give the network absolute coordinates:

```text
Snake = (5, 12)
Food  = (20, 12)
```

The problem with this representation is that the same situation can have completely different coordinates depending on where the snake is on the board.

Instead, the agent gets information relative to its current direction:

```text
Food is ahead
Danger is on the left
```

So if the snake rotates, the representation changes with it.

For example, these two situations might have completely different absolute coordinates:

```text
Snake facing Up
Food is on the Left
```

and:

```text
Snake facing Right
Food is on the Left
```

But from the agent's perspective, both simply mean:

```text
Food -> Left
```

This gives the network a more consistent representation of the problem.

---

# Reward System

The environment gives the agent rewards and penalties based on what happens.

| Event                 |             Reward |
| --------------------- | -----------------: |
| Eating food           |            `+10.0` |
| Collision             |            `-10.0` |
| Moving closer to food | `+0.15` to `+0.30` |
| Moving away from food |            `-0.20` |
| Starvation            |             `-8.0` |

The starvation penalty is applied when the snake goes too many steps without eating.

The idea is to encourage the snake to actually make progress instead of finding a way to survive indefinitely without reaching the food.

---

# Running the Game

## Interactive Mode

Start the game with:

```bash
python main.py
```

Alternatively:

```bash
python fly_snake_game.py
```

The trained model is loaded automatically from:

```text
models/dqn_snake.pth
```

## Controls

| Key            | Action                               |
| -------------- | ------------------------------------ |
| `M`            | Toggle between AI and manual control |
| `F`            | Toggle fast training mode            |
| `O`            | Toggle food proximity heatmap        |
| `S`            | Save the current model               |
| `SPACE`        | Restart the episode                  |
| `UP / DOWN`    | Change simulation speed              |
| `LEFT / RIGHT` | Control the snake in manual mode     |

---

# Headless Training

Training doesn't need the Pygame renderer, so the project also has a headless training mode.

Run:

```bash
python train_headless.py --episodes 300
```

This runs the environment without drawing the game window and is considerably faster than training through the GUI.

The training script reports:

* Episode number
* Score
* Loss
* Epsilon

The model is saved to:

```text
models/dqn_snake.pth
```

---

# Results

The current results compared with the earlier Hebbian approach are:

| Metric        |                    Hebbian Model |                                Double DQN |
| ------------- | -------------------------------: | ----------------------------------------: |
| Average Score |                      ~0–1 fruits |                             ~12–16 fruits |
| High Score    |                         3 fruits |                                44+ fruits |
| Training      |        Did not reliably converge |                   Learned useful behavior |
| Movement      | Random spinning and wall crashes | Better navigation and collision avoidance |

The main improvement was not just the higher score.

The DQN agent started showing behavior that looked like it was actually using the information from its state:

* Moving toward food
* Avoiding nearby obstacles
* Changing direction when necessary
* Learning from previous attempts

---

# Conclusion

This project started as an experiment with a biologically-inspired learning approach and eventually moved toward a more traditional reinforcement learning setup.

The fruit fly approach was interesting, but Snake requires decisions where the result of an action might only become obvious several moves later.

Double DQN provided a better way to handle this through:

* Experience replay
* Discounted future rewards
* Bellman targets
* Epsilon-greedy exploration
* Separate policy and target networks

Another important part of the project was the state representation. Using an ego-centric representation reduced the amount of information the network had to deal with and made similar situations look more consistent from the agent's perspective.
