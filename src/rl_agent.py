"""
Deep Q-Network (DQN) Reinforcement Learning Agent for Snake.
Uses Double DQN with Experience Replay, Target Network, and Bellman Optimization.
"""
import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """Deep Q-Network Architecture"""
    def __init__(self, state_dim=16, action_dim=3):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience Replay Memory Buffer"""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Double Deep Q-Network (DQN) Agent.
    """
    def __init__(self, state_dim=16, action_dim=3, lr=1e-3, gamma=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995

        # PyTorch device (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(capacity=50000)

        # Telemetry for HUD
        self.last_q_values = np.zeros(action_dim, dtype=np.float32)
        self.last_loss = 0.0
        self.total_training_steps = 0

    def select_action(self, state, evaluate=False):
        """
        Select action using epsilon-greedy policy.
        If evaluate=True, chooses greedy action (no random exploration).
        """
        if not evaluate and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_vals = self.policy_net(state_tensor).squeeze().cpu().numpy()
                self.last_q_values = q_vals
            return action

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals = self.policy_net(state_tensor).squeeze().cpu().numpy()
            self.last_q_values = q_vals
            action = int(np.argmax(q_vals))
            return action

    def train_step(self, batch_size=64):
        """Perform one step of Double DQN training with experience replay."""
        if len(self.memory) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # Current Q-values
        curr_q = self.policy_net(states).gather(1, actions)

        # Double DQN target computation:
        # Best action from policy net, evaluated with target net
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.total_training_steps += 1
        self.last_loss = float(loss.item())

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Periodic target network sync
        if self.total_training_steps % 150 == 0:
            self.update_target_network()

        return self.last_loss

    def update_target_network(self):
        """Copies weights from policy_net to target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        """Save network weights and hyperparameters."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_training_steps
        }, filepath)

    def load(self, filepath):
        """Load network weights if available."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            self.total_training_steps = checkpoint.get('total_steps', 0)
            print(f"[DQN] Successfully loaded pre-trained model from {filepath}")
            return True
        return False
