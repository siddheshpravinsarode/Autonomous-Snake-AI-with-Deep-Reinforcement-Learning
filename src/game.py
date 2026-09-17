"""
Snake Game Environment logic: grid management, physics, collision detection, and reward signals.
"""
import math
import random
from src.constants import (
    GRID_COLS, GRID_ROWS, DIR_VECTORS
)
from src.sensors import get_odor_concentration

class SnakeGame:
    def __init__(self, cols=GRID_COLS, rows=GRID_ROWS):
        self.cols = cols
        self.rows = rows
        self.generation = 1
        self.best_score = 0
        self.total_fruits_eaten = 0
        self.scent_following_steps = 0
        self.total_lifetime_steps = 0

        self.last_smell_level = 0.0
        self.particles = []
        self.reset(is_first=True)

    def reset(self, is_first=False):
        if not is_first:
            self.generation += 1
            if self.score > self.best_score:
                self.best_score = self.score

        # Place initial snake in center horizontally moving right
        center_x, center_y = self.cols // 2, self.rows // 2
        self.snake = [
            (center_x, center_y),
            (center_x - 1, center_y),
            (center_x - 2, center_y)
        ]
        self.direction = 1  # 0: Up, 1: Right, 2: Down, 3: Left
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.scent_following_steps = 0
        self.game_over = False
        self.last_reward = 0.0

        self.food = self._spawn_food()
        self.last_smell_level = get_odor_concentration(self.food, self.snake[0][0], self.snake[0][1])
        return self

    def _spawn_food(self):
        while True:
            pos = (random.randint(1, self.cols - 2), random.randint(1, self.rows - 2))
            if pos not in self.snake:
                return pos

    def step(self, action):
        """
        Executes one environment step.
        action:
            0: Turn Left (-90 deg)
            1: Go Straight
            2: Turn Right (+90 deg)
        Returns:
            (reward, game_over, score)
        """
        if self.game_over:
            return 0.0, True, self.score

        self.steps += 1
        self.steps_since_food += 1
        self.total_lifetime_steps += 1

        # 1. Update Direction based on relative action
        if action == 0:    # Turn Left
            self.direction = (self.direction - 1) % 4
        elif action == 2:  # Turn Right
            self.direction = (self.direction + 1) % 4
        # action == 1: Go Straight (no change)

        # 2. Compute New Head Position
        curr_dx, curr_dy = DIR_VECTORS[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + curr_dx, head_y + curr_dy)

        # Calculate pre-move distance to food for progress reward
        old_dist = math.hypot(head_x - self.food[0], head_y - self.food[1])
        new_dist = math.hypot(new_head[0] - self.food[0], new_head[1] - self.food[1])

        # 3. Collision Check (Wall or Self)
        if (new_head[0] < 0 or new_head[0] >= self.cols or
            new_head[1] < 0 or new_head[1] >= self.rows or
            new_head in self.snake):
            self.game_over = True
            reward = -10.0  # Significant collision penalty
            self.last_reward = reward
            return reward, True, self.score

        # 4. Advance Snake Body
        self.snake.insert(0, new_head)

        # Odor gradient evaluation
        new_smell = get_odor_concentration(self.food, new_head[0], new_head[1])
        smell_delta = new_smell - self.last_smell_level
        if smell_delta > 0:
            self.scent_following_steps += 1
        self.last_smell_level = new_smell

        # 5. Food Eating & Starvation
        if new_head == self.food:
            self.score += 1
            self.total_fruits_eaten += 1
            self.steps_since_food = 0
            self.food = self._spawn_food()
            reward = 10.0  # High positive fruit reward
        else:
            self.snake.pop()
            # Starvation / loop prevention (scaled with snake length)
            max_steps_allowed = 120 + len(self.snake) * 10
            if self.steps_since_food > max_steps_allowed:
                self.game_over = True
                reward = -8.0  # Starvation penalty
            else:
                # Chemotaxis & distance shaping: reward moving closer to food/scent
                if new_dist < old_dist:
                    reward = 0.15 + (smell_delta * 5.0)
                else:
                    reward = -0.20

        self.last_reward = reward
        return reward, self.game_over, self.score
