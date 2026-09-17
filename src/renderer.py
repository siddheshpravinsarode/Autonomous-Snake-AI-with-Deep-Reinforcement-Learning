"""
Visual Arena Renderer: Scent / target radiation particles, arena grid, and bioluminescent snake.
"""
import math
import random
import pygame
from src.constants import (
    ARENA_WIDTH, ARENA_HEIGHT, ARENA_MARGIN,
    COLOR_ARENA_BG, COLOR_PANEL_BORDER, COLOR_GRID,
    COLOR_FOOD, COLOR_FOOD_PULSE, COLOR_SNAKE_HEAD,
    COLOR_SNAKE_BODY, DIR_VECTORS
)
from src.sensors import get_food_proximity

class TargetParticle:
    """Animated glowing particle drifting outward from target food."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.3, 1.2)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = random.randint(35, 75)
        self.age = 0
        self.size = random.uniform(1.5, 3.0)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.age += 1
        return self.age < self.lifetime

    def draw(self, surface):
        alpha = max(0, int(160 * (1.0 - (self.age / self.lifetime))))
        radius = int(self.size * (1.0 + (self.age / self.lifetime) * 0.7))
        s = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (255, 100, 160, alpha), (radius + 1, radius + 1), radius)
        surface.blit(s, (int(self.x) - radius, int(self.y) - radius))


def draw_arena(screen, game, show_target_mesh=True):
    """Renders the game arena, target fruit, glowing aura, and snake agent."""
    cell_w = ARENA_WIDTH / game.cols
    cell_h = ARENA_HEIGHT / game.rows

    arena_rect = pygame.Rect(ARENA_MARGIN, ARENA_MARGIN, ARENA_WIDTH, ARENA_HEIGHT)
    pygame.draw.rect(screen, COLOR_ARENA_BG, arena_rect)
    pygame.draw.rect(screen, COLOR_PANEL_BORDER, arena_rect, 2)

    # 1. Target Proximity Diffusion Heatmap Overlay (if enabled)
    if show_target_mesh:
        fx, fy = game.food
        max_rings = 14
        for ring in range(max_rings, 0, -1):
            radius = int((ring * 2.2) * min(cell_w, cell_h))
            prox_val = 1.0 / (1.0 + 0.18 * (ring * 2.0))
            alpha = int(40 * prox_val)
            ring_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(ring_surf, (255, 60, 110, alpha), (radius, radius), radius)
            cx = int(ARENA_MARGIN + (fx + 0.5) * cell_w)
            cy = int(ARENA_MARGIN + (fy + 0.5) * cell_h)
            screen.blit(ring_surf, (cx - radius, cy - radius))

    # 2. Subtle Arena Grid Lines
    for c in range(game.cols + 1):
        gx = ARENA_MARGIN + c * cell_w
        pygame.draw.line(screen, COLOR_GRID, (gx, ARENA_MARGIN), (gx, ARENA_MARGIN + ARENA_HEIGHT))
    for r in range(game.rows + 1):
        gy = ARENA_MARGIN + r * cell_h
        pygame.draw.line(screen, COLOR_GRID, (ARENA_MARGIN, gy), (ARENA_MARGIN + ARENA_WIDTH, gy))

    # 3. Drifting Target Particles
    for p in game.particles:
        p.draw(screen)

    # 4. Target Fruit with Pulsing Glow
    fx, fy = game.food
    food_x = ARENA_MARGIN + fx * cell_w + 2
    food_y = ARENA_MARGIN + fy * cell_h + 2
    food_rect = pygame.Rect(food_x, food_y, cell_w - 4, cell_h - 4)
    pygame.draw.rect(screen, COLOR_FOOD, food_rect, border_radius=6)
    pulse_size = int(3 + 2 * math.sin(game.steps * 0.3))
    center_food_x = int(food_x + (cell_w - 4) / 2)
    center_food_y = int(food_y + (cell_h - 4) / 2)
    pygame.draw.circle(screen, COLOR_FOOD_PULSE, (center_food_x, center_food_y), pulse_size)

    # 5. Snake Body & Head
    for i, (sx, sy) in enumerate(game.snake):
        seg_x = ARENA_MARGIN + sx * cell_w + 1
        seg_y = ARENA_MARGIN + sy * cell_h + 1
        seg_rect = pygame.Rect(seg_x, seg_y, cell_w - 2, cell_h - 2)

        if i == 0:
            # Snake Head
            pygame.draw.rect(screen, COLOR_SNAKE_HEAD, seg_rect, border_radius=5)

            # Eyes (oriented in direction of movement)
            head_cx = seg_x + (cell_w - 2) / 2
            head_cy = seg_y + (cell_h - 2) / 2
            curr_dx, curr_dy = DIR_VECTORS[game.direction]

            eye_col = (15, 20, 30)
            eye_offset_x = -curr_dy * 4 + curr_dx * 3
            eye_offset_y = curr_dx * 4 + curr_dy * 3
            pygame.draw.circle(screen, eye_col, (int(head_cx + eye_offset_x), int(head_cy + eye_offset_y)), 2)
            pygame.draw.circle(screen, eye_col, (int(head_cx - eye_offset_x), int(head_cy - eye_offset_y)), 2)
        else:
            # Gradient body segments
            fade = max(0.2, 1.0 - (i / len(game.snake)))
            body_col = (
                int(COLOR_SNAKE_BODY[0] * fade),
                int(COLOR_SNAKE_BODY[1] * fade + 30 * (1 - fade)),
                int(COLOR_SNAKE_BODY[2] * fade + 20 * (1 - fade))
            )
            pygame.draw.rect(screen, body_col, seg_rect, border_radius=3)
