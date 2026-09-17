import os
import sys

# Windows fix for systemprofile PATH permission issue
path_dirs = os.environ.get("PATH", "").split(os.pathsep)
os.environ["PATH"] = os.pathsep.join([d for d in path_dirs if "systemprofile" not in d.lower()])

import math
import random
import pygame
import numpy as np
import pandas as pd

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fruit Fly Connectome Brain-Controlled AI Agent")
clock = pygame.time.Clock()

# Colors
BG_COLOR = (20, 24, 33)
FLY_COLOR = (255, 204, 0)
FOOD_COLOR = (50, 220, 100)
DANGER_COLOR = (240, 60, 60)
TEXT_COLOR = (240, 240, 240)
NEURON_OFF = (40, 50, 70)
NEURON_ON = (0, 255, 200)

class FlyBrainConnectome:
    """
    Biologically-inspired neural network using Fruit Fly Antennal Lobe -> Kenyon Cell -> MBON architecture
    with Dopaminergic Real-time Plasticity (Hebbian Learning).
    """
    def __init__(self, num_pn=10, num_kc=40, num_mbon=2):
        self.num_pn = num_pn    # Projection Neurons (Sensory inputs: left/right odors)
        self.num_kc = num_kc    # Kenyon Cells (Sparse memory coding layer)
        self.num_mbon = num_mbon # Mushroom Body Output Neurons (Motor decisions: Turn Left / Turn Right)
        
        # 1. Connectome Fixed Synaptic Weights (PN -> KC expansion layer)
        np.random.seed(42)
        self.W_pn_kc = np.random.choice([0, 1], size=(self.num_pn, self.num_kc), p=[0.8, 0.2]).astype(float)
        
        # 2. Plastic Synaptic Weights (KC -> MBON output layer)
        self.W_kc_mbon = np.ones((self.num_kc, self.num_mbon)) * 0.5
        
        # Live firing states for HUD visualization
        self.pn_activity = np.zeros(self.num_pn)
        self.kc_activity = np.zeros(self.num_kc)
        self.mbon_activity = np.zeros(self.num_mbon)

    def forward(self, left_sensor, right_sensor):
        """Forward pass through fly brain connectome"""
        # Encode sensory inputs into PN neurons
        self.pn_activity[:self.num_pn//2] = left_sensor
        self.pn_activity[self.num_pn//2:] = right_sensor
        
        # PN -> KC activation with winner-take-all sparse thresholding (top 10% active KCs)
        raw_kc = np.dot(self.pn_activity, self.W_pn_kc)
        threshold = np.percentile(raw_kc, 90)
        self.kc_activity = (raw_kc > threshold).astype(float) * raw_kc
        
        # KC -> MBON motor output
        self.mbon_activity = np.dot(self.kc_activity, self.W_kc_mbon)
        
        # MBON[0] = Steering bias (Left vs Right), MBON[1] = Forward thrust
        steering = self.mbon_activity[0] - self.mbon_activity[1]
        return steering

    def apply_dopamine_reward(self, reward_signal, learning_rate=0.05):
        """
        Dopaminergic Synaptic Plasticity (Hebbian Learning):
        Updates active KC -> MBON synaptic weights based on reward signal.
        """
        active_kcs = (self.kc_activity > 0).astype(float)
        for m in range(self.num_mbon):
            self.W_kc_mbon[:, m] += learning_rate * reward_signal * active_kcs
            # Keep weights bounded
            self.W_kc_mbon[:, m] = np.clip(self.W_kc_mbon[:, m], 0.0, 2.0)

class FlyAgent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 3.0
        self.brain = FlyBrainConnectome()
        self.score = 0
        self.last_reward = 0

    def sense_environment(self, food_pos):
        """Simulates antennae sensing odor concentrations (Left & Right antennae)"""
        antenna_offset = 0.4
        antenna_len = 20
        
        # Left antenna position
        lx = self.x + math.cos(self.angle - antenna_offset) * antenna_len
        ly = self.y + math.sin(self.angle - antenna_offset) * antenna_len
        dist_left = math.hypot(food_pos[0] - lx, food_pos[1] - ly)
        left_signal = max(0, 1000 - dist_left) / 1000.0

        # Right antenna position
        rx = self.x + math.cos(self.angle + antenna_offset) * antenna_len
        ry = self.y + math.sin(self.angle + antenna_offset) * antenna_len
        dist_right = math.hypot(food_pos[0] - rx, food_pos[1] - ry)
        right_signal = max(0, 1000 - dist_right) / 1000.0

        return left_signal, right_signal

    def update(self, food_pos):
        l_sig, r_sig = self.sense_environment(food_pos)
        
        # Forward pass through fly brain connectome
        steering = self.brain.forward(l_sig, r_sig)
        
        # Update orientation & position
        self.angle += steering * 0.1
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        
        # Keep inside screen boundaries
        self.x = max(20, min(WIDTH - 250, self.x))
        self.y = max(20, min(HEIGHT - 20, self.y))
        
        # Check food capture
        dist_to_food = math.hypot(food_pos[0] - self.x, food_pos[1] - self.y)
        if dist_to_food < 25:
            self.score += 1
            self.brain.apply_dopamine_reward(reward_signal=1.0) # Reward active synapses
            self.last_reward = 1.0
            return True # Food eaten!
        else:
            self.last_reward = 0.0
            return False

    def draw(self, surface):
        # Draw fly body
        pygame.draw.circle(surface, FLY_COLOR, (int(self.x), int(self.y)), 10)
        # Draw head / direction indicator
        hx = self.x + math.cos(self.angle) * 15
        hy = self.y + math.sin(self.angle) * 15
        pygame.draw.line(surface, (255, 255, 255), (self.x, self.y), (hx, hy), 3)

def main():
    food_x = random.randint(100, WIDTH - 350)
    food_y = random.randint(100, HEIGHT - 100)
    fly = FlyAgent(WIDTH // 3, HEIGHT // 2)

    font = pygame.font.SysFont("Arial", 16)
    title_font = pygame.font.SysFont("Arial", 20, bold=True)

    running = True
    while running:
        clock.tick(60)
        screen.fill(BG_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Move food to mouse click position
                mx, my = pygame.mouse.get_pos()
                if mx < WIDTH - 250:
                    food_x, food_y = mx, my

        # Update Fly Agent
        eaten = fly.update((food_x, food_y))
        if eaten:
            food_x = random.randint(100, WIDTH - 350)
            food_y = random.randint(100, HEIGHT - 100)

        # Draw Food Plume / Reward Target
        pygame.draw.circle(screen, (50, 220, 100, 50), (food_x, food_y), 40)
        pygame.draw.circle(screen, FOOD_COLOR, (food_x, food_y), 12)

        # Draw Fly
        fly.draw(screen)

        # ==========================================
        # REAL-TIME CONNECTOME NEURAL HUD PANEL
        # ==========================================
        panel_x = WIDTH - 240
        pygame.draw.rect(screen, (30, 36, 50), (panel_x - 10, 10, 240, HEIGHT - 20), border_radius=8)
        
        screen.blit(title_font.render("FLY BRAIN HUD", True, (0, 255, 200)), (panel_x, 20))
        screen.blit(font.render(f"Score / Food Captured: {fly.score}", True, TEXT_COLOR), (panel_x, 50))
        screen.blit(font.render(f"Dopamine Reward: {fly.last_reward}", True, (255, 215, 0)), (panel_x, 75))

        # Render Sensory Neurons (Projection Neurons)
        screen.blit(font.render("Antennal Lobe PNs (Sensory):", True, TEXT_COLOR), (panel_x, 110))
        for i, val in enumerate(fly.brain.pn_activity):
            col = NEURON_ON if val > 0.1 else NEURON_OFF
            px = panel_x + (i % 5) * 20
            py = 135 + (i // 5) * 20
            pygame.draw.circle(screen, col, (px + 10, py + 10), 7)

        # Render Kenyon Cells (KC Sparse Layer)
        screen.blit(font.render("Mushroom Body KCs (Memory):", True, TEXT_COLOR), (panel_x, 185))
        for i, val in enumerate(fly.brain.kc_activity):
            col = (0, 255, 200) if val > 0 else NEURON_OFF
            px = panel_x + (i % 10) * 20
            py = 210 + (i // 10) * 20
            pygame.draw.circle(screen, col, (px + 7, py + 7), 5)

        # Instructions
        screen.blit(font.render("Click anywhere to move food!", True, (180, 180, 180)), (panel_x, 340))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
