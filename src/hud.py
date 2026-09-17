"""
Reinforcement Learning HUD Dashboard.
Provides real-time telemetry: Q-values, Bellman loss, replay memory,
epsilon exploration rate, radar hazard detectors, and game statistics.
"""
import pygame
import numpy as np
from src.constants import (
    COLOR_HUD_BG, COLOR_PANEL_BORDER, COLOR_TEXT, COLOR_TEXT_DIM,
    COLOR_REWARD_POS, COLOR_REWARD_NEG, COLOR_ACCENT, COLOR_RL_BLUE,
    ACTION_NAMES
)
from src.sensors import get_food_proximity, extract_rl_state

class VisualizerHUD:
    def __init__(self, surface, x, y, width, height):
        self.surface = surface
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.title_font = pygame.font.SysFont("Trebuchet MS", 18, bold=True)
        self.section_font = pygame.font.SysFont("Trebuchet MS", 13, bold=True)
        self.font = pygame.font.SysFont("Consolas", 11)
        self.stat_font = pygame.font.SysFont("Consolas", 12, bold=True)

    def draw(self, game, rl_agent, auto_mode=True, speed=14, show_target_mesh=True):
        panel_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(self.surface, COLOR_HUD_BG, panel_rect)
        pygame.draw.rect(self.surface, COLOR_PANEL_BORDER, panel_rect, 2)

        px = self.x + 14
        py = self.y + 12

        # 1. Header
        title_surf = self.title_font.render("DQN REINFORCEMENT LEARNING", True, COLOR_ACCENT)
        self.surface.blit(title_surf, (px, py))
        py += 22

        mode_str = "AUTONOMOUS DQN AGENT" if auto_mode else "MANUAL HUMAN OVERRIDE"
        mode_col = (130, 180, 255) if auto_mode else (255, 200, 50)
        self.surface.blit(self.font.render(f"Mode: {mode_str}", True, mode_col), (px, py))
        py += 22

        # 2. Performance & Episode Metrics
        pygame.draw.line(self.surface, COLOR_PANEL_BORDER, (px, py), (self.x + self.width - 14, py), 1)
        py += 8

        self.surface.blit(self.section_font.render("EPISODE & PERFORMANCE TELEMETRY", True, COLOR_TEXT), (px, py))
        py += 18

        stats = [
            f"Episode / Gen  : #{game.generation} (Best Score: {game.best_score})",
            f"Fruits Eaten   : {game.score} (Total Lifetime: {game.total_fruits_eaten})",
            f"Steps Survived : {game.steps} | Simulation Speed: {speed} FPS",
            f"Step Reward    : {game.last_reward:+.2f} | Bellman Loss: {rl_agent.last_loss:.4f}",
            f"Replay Memory  : {len(rl_agent.memory):,}/50,000 transitions",
            f"Exploration    : {rl_agent.epsilon*100:.1f}% (Epsilon)"
        ]
        for s in stats:
            self.surface.blit(self.font.render(s, True, COLOR_TEXT_DIM), (px, py))
            py += 15

        # Reward Bar Meter
        py += 4
        dopamine_val = game.last_reward
        center_bar_x = px + (self.width - 28) // 2
        pygame.draw.rect(self.surface, (30, 38, 52), (px, py, self.width - 28, 8), border_radius=4)
        pygame.draw.line(self.surface, (100, 110, 130), (center_bar_x, py), (center_bar_x, py + 8), 1)
        if dopamine_val != 0:
            fill = int((abs(dopamine_val) / 10.0) * ((self.width - 28) // 2))
            fill = min(fill, (self.width - 28) // 2)
            col = COLOR_REWARD_POS if dopamine_val > 0 else COLOR_REWARD_NEG
            if dopamine_val > 0:
                bar = pygame.Rect(center_bar_x, py, fill, 8)
            else:
                bar = pygame.Rect(center_bar_x - fill, py, fill, 8)
            pygame.draw.rect(self.surface, col, bar, border_radius=3)
        py += 16

        # 3. Real-Time Q-Values Bar Graph
        pygame.draw.line(self.surface, COLOR_PANEL_BORDER, (px, py), (self.x + self.width - 14, py), 1)
        py += 8

        self.surface.blit(self.section_font.render("ACTION-VALUE FUNCTION Q(s, a)", True, (140, 180, 255)), (px, py))
        py += 18

        q_vals = rl_agent.last_q_values
        max_q_idx = int(np.argmax(q_vals)) if len(q_vals) > 0 else 0

        action_bar_w = self.width - 150
        min_q = min(-1.0, float(np.min(q_vals)))
        max_q = max(1.0, float(np.max(q_vals)))
        q_range = max(0.1, max_q - min_q)

        for i, name in enumerate(ACTION_NAMES):
            q = q_vals[i] if i < len(q_vals) else 0.0
            is_best = (i == max_q_idx)

            lbl_col = (0, 255, 200) if is_best else COLOR_TEXT_DIM
            self.surface.blit(self.font.render(f"{name:<11}", True, lbl_col), (px, py))

            bx = px + 95
            pygame.draw.rect(self.surface, (30, 38, 52), (bx, py, action_bar_w, 14), border_radius=3)

            norm_fill = (q - min_q) / q_range
            fill_w = int(norm_fill * action_bar_w)
            bar_col = (60, 230, 160) if is_best else (70, 100, 150)
            if fill_w > 0:
                pygame.draw.rect(self.surface, bar_col, (bx, py, fill_w, 14), border_radius=3)

            if is_best:
                pygame.draw.rect(self.surface, (200, 255, 230), (bx, py, action_bar_w, 14), 1, border_radius=3)

            val_surf = self.font.render(f"{q:+.2f}", True, (255, 255, 255) if is_best else (160, 170, 190))
            self.surface.blit(val_surf, (bx + action_bar_w + 8, py))
            py += 22

        # 4. Ego-Centric Sensor Radar Panel
        py += 8
        pygame.draw.line(self.surface, COLOR_PANEL_BORDER, (px, py), (self.x + self.width - 14, py), 1)
        py += 8

        self.surface.blit(self.section_font.render("EGO-CENTRIC HAZARD & TARGET RADAR", True, COLOR_TEXT), (px, py))
        py += 18

        state = extract_rl_state(game)
        danger_fwd = state[0] > 0.5
        danger_lft = state[1] > 0.5
        danger_rgt = state[2] > 0.5

        food_ahead = state[7] > 0.5
        food_left = state[8] > 0.5
        food_right = state[9] > 0.5
        food_behind = state[10] > 0.5

        # Radar Indicators
        def draw_badge(text, active, x_pos, y_pos, active_col=(255, 65, 65)):
            col = active_col if active else (40, 50, 70)
            text_col = (255, 255, 255) if active else (110, 120, 140)
            rect = pygame.Rect(x_pos, y_pos, 74, 18)
            pygame.draw.rect(self.surface, col, rect, border_radius=4)
            pygame.draw.rect(self.surface, (70, 80, 100), rect, 1, border_radius=4)
            lbl = self.font.render(text, True, text_col)
            self.surface.blit(lbl, (x_pos + 6, y_pos + 3))

        self.surface.blit(self.font.render("Collision Hazards :", True, (255, 180, 100)), (px, py + 2))
        draw_badge("LFT DANGER", danger_lft, px + 120, py)
        draw_badge("FWD DANGER", danger_fwd, px + 200, py)
        draw_badge("RGT DANGER", danger_rgt, px + 280, py)
        py += 24

        self.surface.blit(self.font.render("Food Direction    :", True, (100, 220, 255)), (px, py + 2))
        draw_badge("AHEAD", food_ahead, px + 120, py, active_col=(0, 180, 130))
        draw_badge("LEFT", food_left, px + 200, py, active_col=(0, 180, 130))
        draw_badge("RIGHT", food_right, px + 280, py, active_col=(0, 180, 130))
        draw_badge("BEHIND", food_behind, px + 360, py, active_col=(0, 180, 130))
        py += 28

        # 5. Network Architecture Summary Box
        box_rect = pygame.Rect(px, py, self.width - 28, 90)
        pygame.draw.rect(self.surface, (25, 32, 46), box_rect, border_radius=6)
        pygame.draw.rect(self.surface, COLOR_PANEL_BORDER, box_rect, 1)

        self.surface.blit(self.section_font.render("Deep Q-Network Specs:", True, (200, 215, 240)), (px + 10, py + 6))
        specs = [
            "- Model Topology  : 16 Input Features -> 128 ReLU -> 128 ReLU -> 3 Q-Values",
            "- Target Network  : Synced every 150 training updates (Double DQN)",
            "- Optimizer / Loss: Adam (lr=1e-3, Huber Smooth-L1 loss, Gradient Clip: 5.0)",
            "- Replay Strategy : Uniform random mini-batch (batch_size=64)"
        ]
        for idx, sp in enumerate(specs):
            self.surface.blit(self.font.render(sp, True, COLOR_TEXT_DIM), (px + 10, py + 24 + idx * 15))

        # 6. Controls & Hotkeys Footer
        py_bottom = self.y + self.height - 72
        pygame.draw.line(self.surface, COLOR_PANEL_BORDER, (px, py_bottom), (self.x + self.width - 14, py_bottom), 1)
        py_bottom += 8

        self.surface.blit(self.section_font.render("SIMULATION CONTROLS & SHORTCUTS", True, COLOR_TEXT), (px, py_bottom))
        py_bottom += 18
        controls = [
            "[M] Toggle AI / Manual Steering       [F] Toggle Fast 140 FPS Mode",
            "[O] Toggle Target Proximity Overlay   [S] Save Model Weights",
            "[SPACE] Restart Episode               [UP / DOWN] Speed +/-"
        ]
        for c in controls:
            self.surface.blit(self.font.render(c, True, (150, 165, 190)), (px, py_bottom))
            py_bottom += 14
