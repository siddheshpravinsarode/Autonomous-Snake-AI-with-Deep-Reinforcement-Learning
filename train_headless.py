"""
Headless Reinforcement Learning Trainer.
Trains the Double DQN Agent rapidly without Pygame graphics rendering.
"""
import os
import sys
import time
import argparse
import numpy as np

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.game import SnakeGame
from src.sensors import extract_rl_state
from src.rl_agent import DQNAgent

def train(episodes=300, batch_size=64, save_path="models/dqn_snake.pth"):
    print(f"==================================================")
    print(f" Starting Headless DQN Training for {episodes} Episodes")
    print(f" Target Model File: {save_path}")
    print(f"==================================================")

    agent = DQNAgent(state_dim=16, action_dim=3, lr=1e-3, gamma=0.95)
    # Check if pre-existing model exists
    if os.path.exists(save_path):
        agent.load(save_path)

    game = SnakeGame()
    start_time = time.time()

    scores = []
    best_score = 0

    for ep in range(1, episodes + 1):
        game.reset()
        state = extract_rl_state(game)
        ep_reward = 0.0
        losses = []

        while not game.game_over:
            # Select action
            action = agent.select_action(state)

            # Step environment
            reward, done, score = game.step(action)
            ep_reward += reward

            # Next state
            next_state = extract_rl_state(game)

            # Store in replay buffer
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state

            # Train network
            loss = agent.train_step(batch_size=batch_size)
            if loss > 0:
                losses.append(loss)

        scores.append(game.score)
        if game.score > best_score:
            best_score = game.score

        if ep % 25 == 0 or ep == episodes:
            avg_score = np.mean(scores[-25:])
            avg_loss = np.mean(losses) if losses else 0.0
            elapsed = time.time() - start_time
            fps = game.total_lifetime_steps / max(1, elapsed)
            print(f"[Episode {ep:04d}/{episodes}] Avg Score: {avg_score:.2f} | Best: {best_score} | "
                  f"Epsilon: {agent.epsilon:.3f} | Loss: {avg_loss:.4f} | Speed: {fps:.0f} steps/s")

    # Save model
    agent.save(save_path)
    total_time = time.time() - start_time
    print(f"\n[Training Complete] Finished in {total_time:.2f}s.")
    print(f"[Model Saved] -> {save_path}")
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Double DQN Trainer for FlySnake")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--batch_size", type=int, default=64, help="Replay buffer batch size")
    parser.add_argument("--output", type=str, default="models/dqn_snake.pth", help="Path to save model")
    args = parser.parse_args()

    train(episodes=args.episodes, batch_size=args.batch_size, save_path=args.output)
