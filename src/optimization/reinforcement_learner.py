#!/usr/bin/env python
"""
Reinforcement Learning Module

This module implements reinforcement learning for dynamic adaptation of trading
strategies using Deep Q-Networks (DQN) and policy gradients.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
import random
import logging
from datetime import datetime

# Define experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """Deep Q-Network for action-value function approximation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Initialize DQN network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(state)

class PolicyNetwork(nn.Module):
    """Policy network for continuous action space."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Initialize policy network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output mean and log_std for action distribution
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.network(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        return mean, log_std

class ReinforcementLearner:
    """Implements reinforcement learning for strategy adaptation."""
    
    def __init__(self,
                state_dim: int,
                action_dim: int,
                learning_type: str = 'dqn',
                hidden_dim: int = 64,
                learning_rate: float = 1e-3,
                gamma: float = 0.99,
                buffer_size: int = 10000,
                batch_size: int = 64,
                target_update: int = 10,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.01,
                epsilon_decay: float = 0.995):
        """
        Initialize the reinforcement learner.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_type: Type of RL algorithm ('dqn' or 'policy')
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate
            gamma: Discount factor
            buffer_size: Size of replay buffer
            batch_size: Training batch size
            target_update: Frequency of target network update
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of exploration decay
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_type = learning_type
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Initialize networks
        if learning_type == 'dqn':
            self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim)
            self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
            
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Training stats
        self.training_stats = []
        self.episode_rewards = []
        
        self.logger = logging.getLogger(__name__)
        
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if self.learning_type == 'dqn':
            return self._select_action_dqn(state)
        else:
            return self._select_action_policy(state)
            
    def _select_action_dqn(self, state: np.ndarray) -> int:
        """Select action using DQN with epsilon-greedy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
            
    def _select_action_policy(self, state: np.ndarray) -> np.ndarray:
        """Select action using policy network."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, log_std = self.policy_net(state_tensor)
            std = torch.exp(log_std)
            
            # Sample action from normal distribution
            normal = Normal(mean, std)
            action = normal.sample()
            
            return action.numpy().squeeze()
            
    def store_experience(self,
                       state: np.ndarray,
                       action: np.ndarray,
                       reward: float,
                       next_state: np.ndarray,
                       done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.append(
            Experience(state, action, reward, next_state, done)
        )
        
    def train(self) -> Dict[str, float]:
        """
        Train the agent using stored experiences.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
            
        if self.learning_type == 'dqn':
            return self._train_dqn()
        else:
            return self._train_policy()
            
    def _train_dqn(self) -> Dict[str, float]:
        """Train DQN agent."""
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        batch = Experience(*zip(*batch))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.LongTensor(batch.action)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.FloatTensor(batch.done)
        
        # Compute current Q values
        current_q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0]
            target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
            
        # Compute loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if len(self.training_stats) % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Update exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        metrics = {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'avg_q_value': current_q.mean().item()
        }
        
        self.training_stats.append(metrics)
        return metrics
    
    def _train_policy(self) -> Dict[str, float]:
        """Train policy gradient agent."""
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        batch = Experience(*zip(*batch))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.FloatTensor(np.array(batch.action))
        reward_batch = torch.FloatTensor(batch.reward)
        
        # Get action distributions
        mean, log_std = self.policy_net(state_batch)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        
        # Compute log probabilities
        log_probs = normal.log_prob(action_batch).sum(dim=1)
        
        # Compute loss
        loss = -(log_probs * reward_batch).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        metrics = {
            'loss': loss.item(),
            'avg_reward': reward_batch.mean().item(),
            'policy_std': std.mean().item()
        }
        
        self.training_stats.append(metrics)
        return metrics
    
    def save_model(self, path: str):
        """Save model parameters."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict() if self.learning_type == 'dqn' else None,
            'optimizer': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'episode_rewards': self.episode_rewards
        }, path)
        
    def load_model(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        if self.learning_type == 'dqn' and checkpoint['target_net']:
            self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_stats = checkpoint['training_stats']
        self.episode_rewards = checkpoint['episode_rewards']
        
    def get_training_stats(self) -> pd.DataFrame:
        """Get training statistics as DataFrame."""
        return pd.DataFrame(self.training_stats)
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress."""
        import matplotlib.pyplot as plt
        
        stats = self.get_training_stats()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot loss
        stats['loss'].plot(ax=ax1)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot rewards
        if self.episode_rewards:
            pd.Series(self.episode_rewards).plot(ax=ax2)
            ax2.set_title('Episode Rewards')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Total Reward')
            ax2.grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        plt.close()