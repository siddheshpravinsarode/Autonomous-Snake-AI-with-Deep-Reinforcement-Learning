"""
Sensory & Spatial State Perception for Reinforcement Learning Snake.
Converts the raw game grid into an ego-centric, rotation-invariant observation vector.
"""
import math
import numpy as np
from src.constants import DIR_VECTORS

def get_food_proximity(food_pos, x, y):
    """
    Computes an inverse-distance proximity field to the target food:
    Returns values between 0.0 (far away) and 1.0 (directly on food).
    """
    dist = math.hypot(x - food_pos[0], y - food_pos[1])
    return float(1.0 / (1.0 + 0.18 * dist))

# Backward-compatible alias
get_odor_concentration = get_food_proximity


def extract_rl_state(game):
    """
    Constructs a 16-dimensional ego-centric state vector for the DQN agent.
    
    Why ego-centric?
    By sensing danger and targets relative to the snake's head rather than
    absolute compass coordinates, the state space becomes rotation-invariant.
    This allows the DQN agent to generalize across all headings and learn
    in minutes instead of days.

    State Vector (16 Features):
      [0]  Danger Straight : 1.0 if stepping forward causes collision, else 0.0
      [1]  Danger Left     : 1.0 if turning left causes collision, else 0.0
      [2]  Danger Right    : 1.0 if turning right causes collision, else 0.0
      [3]  Direction Up    : 1.0 if currently moving Up, else 0.0
      [4]  Direction Right : 1.0 if currently moving Right, else 0.0
      [5]  Direction Down  : 1.0 if currently moving Down, else 0.0
      [6]  Direction Left  : 1.0 if currently moving Left, else 0.0
      [7]  Food Ahead      : 1.0 if food is in the forward hemisphere, else 0.0
      [8]  Food Left       : 1.0 if food is to the left of heading, else 0.0
      [9]  Food Right      : 1.0 if food is to the right of heading, else 0.0
      [10] Food Behind     : 1.0 if food is behind the snake, else 0.0
      [11] Food Proximity  : Inverse distance to food at head location (0.0 - 1.0)
      [12] Left Sensor     : Proximity at lateral left sensor offset
      [13] Right Sensor    : Proximity at lateral right sensor offset
      [14] Progress Trend  : 1.0 if last move got closer to food, 0.0 if further
      [15] Normalized Dist : Direct Euclidean distance to food / max grid diagonal
    """
    head_x, head_y = game.snake[0]
    food_x, food_y = game.food

    curr_dx, curr_dy = DIR_VECTORS[game.direction]
    left_dir = (game.direction - 1) % 4
    left_dx, left_dy = DIR_VECTORS[left_dir]
    right_dir = (game.direction + 1) % 4
    right_dx, right_dy = DIR_VECTORS[right_dir]

    def is_collision(nx, ny):
        # Wall boundary check
        if nx < 0 or nx >= game.cols or ny < 0 or ny >= game.rows:
            return 1.0
        # Self-body collision check (excluding tail tip which moves away next step)
        if (nx, ny) in game.snake[:-1]:
            return 1.0
        return 0.0

    # 1. Immediate 1-step hazard detection
    danger_straight = is_collision(head_x + curr_dx, head_y + curr_dy)
    danger_left = is_collision(head_x + left_dx, head_y + left_dy)
    danger_right = is_collision(head_x + right_dx, head_y + right_dy)

    # 2. Current heading (one-hot)
    dir_up = 1.0 if game.direction == 0 else 0.0
    dir_right = 1.0 if game.direction == 1 else 0.0
    dir_down = 1.0 if game.direction == 2 else 0.0
    dir_left = 1.0 if game.direction == 3 else 0.0

    # 3. Target food direction relative to current heading
    v_food_x = food_x - head_x
    v_food_y = food_y - head_y

    # Longitudinal projection (ahead vs behind)
    fwd_proj = v_food_x * curr_dx + v_food_y * curr_dy
    # Lateral projection (left vs right)
    lat_proj = v_food_x * right_dx + v_food_y * right_dy

    food_ahead = 1.0 if fwd_proj > 0 else 0.0
    food_behind = 1.0 if fwd_proj < 0 else 0.0
    food_right = 1.0 if lat_proj > 0 else 0.0
    food_left = 1.0 if lat_proj < 0 else 0.0

    # 4. Proximity field values
    head_prox = get_food_proximity(game.food, head_x, head_y)
    left_prox = get_food_proximity(game.food, head_x + curr_dx + left_dx * 0.7, head_y + curr_dy + left_dy * 0.7)
    right_prox = get_food_proximity(game.food, head_x + curr_dx + right_dx * 0.7, head_y + curr_dy + right_dy * 0.7)

    # Progress trend: did the last step bring us closer?
    delta_prox = head_prox - game.last_smell_level
    progress_trend = float(np.clip(0.5 + delta_prox * 8.0, 0.0, 1.0))

    # Normalized distance to food
    max_diag = math.hypot(game.cols, game.rows)
    dist_food = math.hypot(v_food_x, v_food_y) / max_diag

    state = np.array([
        danger_straight,
        danger_left,
        danger_right,
        dir_up,
        dir_right,
        dir_down,
        dir_left,
        food_ahead,
        food_left,
        food_right,
        food_behind,
        head_prox,
        left_prox,
        right_prox,
        progress_trend,
        dist_food
    ], dtype=np.float32)

    return state
