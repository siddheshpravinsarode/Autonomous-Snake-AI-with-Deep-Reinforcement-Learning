"""
Game Constants and Configuration for Deep Reinforcement Learning Snake.
"""
import os

# Windows fix for systemprofile PATH permission issue if present
path_dirs = os.environ.get("PATH", "").split(os.pathsep)
os.environ["PATH"] = os.pathsep.join([d for d in path_dirs if "systemprofile" not in d.lower()])

# Display Dimensions
SCREEN_WIDTH = 1180
SCREEN_HEIGHT = 720
ARENA_WIDTH = 680
ARENA_HEIGHT = 680
ARENA_MARGIN = 20
HUD_WIDTH = SCREEN_WIDTH - ARENA_WIDTH - (ARENA_MARGIN * 2)

# Grid size
GRID_COLS = 28
GRID_ROWS = 28

# Direction vectors: 0: Up, 1: Right, 2: Down, 3: Left
DIR_UP = 0
DIR_RIGHT = 1
DIR_DOWN = 2
DIR_LEFT = 3
DIR_VECTORS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# Actions relative to heading:
# 0: Turn Left (-90 deg), 1: Go Straight (0 deg), 2: Turn Right (+90 deg)
ACTION_LEFT = 0
ACTION_STRAIGHT = 1
ACTION_RIGHT = 2
ACTION_NAMES = ["TURN LEFT", "STRAIGHT", "TURN RIGHT"]

# Cyberpunk / Bioluminescent Palette
COLOR_BG = (12, 15, 22)
COLOR_ARENA_BG = (16, 20, 30)
COLOR_GRID = (25, 32, 46)
COLOR_SNAKE_HEAD = (0, 255, 190)
COLOR_SNAKE_BODY = (0, 185, 135)
COLOR_FOOD = (255, 55, 95)
COLOR_FOOD_PULSE = (255, 120, 150)
COLOR_HUD_BG = (20, 25, 36)
COLOR_PANEL_BORDER = (40, 52, 75)
COLOR_TEXT = (235, 240, 250)
COLOR_TEXT_DIM = (135, 145, 168)
COLOR_ACCENT = (0, 230, 255)
COLOR_RL_BLUE = (80, 140, 255)
COLOR_REWARD_POS = (45, 235, 120)
COLOR_REWARD_NEG = (245, 65, 65)
