"""
Deep Reinforcement Learning Snake Entrypoint.
Modular components in src/:
- src/constants.py: Game dimensions, colors, and key configs
- src/sensors.py: Spatial state perception & ego-centric radar
- src/game.py: Core SnakeGame environment physics and rules
- src/rl_agent.py: Double Deep Q-Network (DQN) PyTorch RL agent
- src/renderer.py: Arena rendering, target particles, and glowing visuals
- src/hud.py: Real-time telemetry, Q-value graphs, and sensory HUD
- train_headless.py: Rapid headless RL trainer
"""
from main import main

if __name__ == "__main__":
    main()
