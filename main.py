"""
Main GUI Application for Deep Reinforcement Learning (DQN) Snake AI.
"""
import os
import sys
import time
import random
import pygame

# Add workspace root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, ARENA_WIDTH, ARENA_HEIGHT,
    ARENA_MARGIN, HUD_WIDTH, COLOR_BG
)
from src.game import SnakeGame
from src.sensors import extract_rl_state
from src.rl_agent import DQNAgent
from src.renderer import draw_arena, TargetParticle
from src.hud import VisualizerHUD

def main():
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Deep Reinforcement Learning (DQN) Snake AI")
    clock = pygame.time.Clock()

    game = SnakeGame()
    hud_x = ARENA_WIDTH + ARENA_MARGIN * 2
    hud = VisualizerHUD(screen, hud_x, ARENA_MARGIN, HUD_WIDTH, ARENA_HEIGHT)

    # RL Agent
    rl_agent = DQNAgent(state_dim=16, action_dim=3, lr=1e-3, gamma=0.95)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "dqn_snake.pth")
    if os.path.exists(model_path):
        rl_agent.load(model_path)
        rl_agent.epsilon = 0.05
    else:
        print("[Notice] No pre-trained model found at models/dqn_snake.pth. Starting fresh.")

    # State & Modes
    auto_mode = True
    fast_mode = False
    show_target_mesh = True
    simulation_speed = 14
    running = True

    manual_action = None

    while running:
        current_fps = 140 if fast_mode else simulation_speed
        clock.tick(current_fps)
        manual_action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.reset()
                elif event.key == pygame.K_m:
                    auto_mode = not auto_mode
                elif event.key == pygame.K_f:
                    fast_mode = not fast_mode
                elif event.key == pygame.K_o:
                    show_target_mesh = not show_target_mesh
                elif event.key == pygame.K_s:
                    rl_agent.save(model_path)
                    print(f"[DQN] Manually saved model to {model_path}")
                elif event.key == pygame.K_UP:
                    simulation_speed = min(80, simulation_speed + 4)
                elif event.key == pygame.K_DOWN:
                    simulation_speed = max(2, simulation_speed - 4)
                elif event.key == pygame.K_LEFT:
                    manual_action = 0
                elif event.key == pygame.K_RIGHT:
                    manual_action = 2

        # Step Simulation
        if not game.game_over:
            # Drifting glowing particles from target food
            cell_w = ARENA_WIDTH / game.cols
            cell_h = ARENA_HEIGHT / game.rows
            fx, fy = game.food
            if random.random() < 0.75:
                px = (fx + 0.5) * cell_w + ARENA_MARGIN
                py = (fy + 0.5) * cell_h + ARENA_MARGIN
                game.particles.append(TargetParticle(px, py))
            game.particles = [p for p in game.particles if p.update()]

            if auto_mode:
                # Deep Q-Network Agent Decision
                curr_state = extract_rl_state(game)
                # Select action (greedy exploitation when watching at normal speed)
                action = rl_agent.select_action(curr_state, evaluate=(not fast_mode and rl_agent.epsilon <= 0.05))
                reward, done, score = game.step(action)
                next_state = extract_rl_state(game)

                # Online continuous learning with experience replay
                rl_agent.memory.push(curr_state, action, reward, next_state, done)
                rl_agent.train_step(batch_size=64)

            else:
                # Manual Keyboard Steering
                act = manual_action if manual_action is not None else 1
                game.step(act)

        else:
            # Episode Ended
            if not fast_mode:
                time.sleep(0.18)

            # Auto-save weights if new high score is achieved
            if game.score >= game.best_score and game.score > 0:
                rl_agent.save(model_path)

            game.reset()

        # Render Frame
        screen.fill(COLOR_BG)
        draw_arena(screen, game, show_target_mesh=show_target_mesh)
        hud.draw(
            game,
            rl_agent=rl_agent,
            auto_mode=auto_mode,
            speed=current_fps,
            show_target_mesh=show_target_mesh
        )
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
