#!/usr/bin/env python
"""Unit tests for reinforcement learning."""

import unittest
import numpy as np
import pandas as pd
import torch
from src.optimization.reinforcement_learner import ReinforcementLearner

class TestReinforcementLearner(unittest.TestCase):
    def setUp(self):
        """Set up test data and learner."""
        # Define state and action dimensions
        self.state_dim = 10
        self.action_dim = 3
        
        # Initialize DQN learner
        self.dqn_learner = ReinforcementLearner(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            learning_type='dqn',
            hidden_dim=32,
            buffer_size=1000
        )
        
        # Initialize Policy learner
        self.policy_learner = ReinforcementLearner(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            learning_type='policy',
            hidden_dim=32,
            buffer_size=1000
        )
        
        # Create sample data
        self.sample_state = np.random.normal(0, 1, self.state_dim)
        self.sample_next_state = np.random.normal(0, 1, self.state_dim)
        self.sample_reward = 1.0
        self.sample_done = False
        
    def test_initialization(self):
        """Test learner initialization."""
        # Test DQN learner
        self.assertEqual(self.dqn_learner.state_dim, self.state_dim)
        self.assertEqual(self.dqn_learner.action_dim, self.action_dim)
        self.assertEqual(self.dqn_learner.learning_type, 'dqn')
        
        # Test policy learner
        self.assertEqual(self.policy_learner.state_dim, self.state_dim)
        self.assertEqual(self.policy_learner.action_dim, self.action_dim)
        self.assertEqual(self.policy_learner.learning_type, 'policy')
        
        # Test network initialization
        self.assertIsNotNone(self.dqn_learner.policy_net)
        self.assertIsNotNone(self.dqn_learner.target_net)
        self.assertIsNotNone(self.policy_learner.policy_net)
        
    def test_dqn_action_selection(self):
        """Test DQN action selection."""
        # Test epsilon-greedy exploration
        self.dqn_learner.epsilon = 0.0  # Force exploitation
        action = self.dqn_learner.select_action(self.sample_state)
        
        # Check action validity
        self.assertIsInstance(action, (int, np.integer))
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
        
        # Test with different epsilon values
        self.dqn_learner.epsilon = 1.0  # Force exploration
        actions = [self.dqn_learner.select_action(self.sample_state) 
                  for _ in range(100)]
        
        # Check exploration distribution
        self.assertTrue(len(set(actions)) > 1)  # Should have different actions
        
    def test_policy_action_selection(self):
        """Test policy gradient action selection."""
        action = self.policy_learner.select_action(self.sample_state)
        
        # Check action dimensions
        self.assertEqual(action.shape, (self.action_dim,))
        
        # Check action values are reasonable
        self.assertTrue(np.all(np.isfinite(action)))
        
    def test_experience_storage(self):
        """Test experience storage in replay buffer."""
        # Store experience in DQN learner
        action = self.dqn_learner.select_action(self.sample_state)
        self.dqn_learner.store_experience(
            self.sample_state,
            action,
            self.sample_reward,
            self.sample_next_state,
            self.sample_done
        )
        
        # Check buffer content
        self.assertEqual(len(self.dqn_learner.replay_buffer), 1)
        experience = self.dqn_learner.replay_buffer[0]
        
        np.testing.assert_array_equal(experience.state, self.sample_state)
        self.assertEqual(experience.action, action)
        self.assertEqual(experience.reward, self.sample_reward)
        np.testing.assert_array_equal(experience.next_state, self.sample_next_state)
        self.assertEqual(experience.done, self.sample_done)
        
    def test_dqn_training(self):
        """Test DQN training step."""
        # Fill buffer with random experiences
        for _ in range(self.dqn_learner.batch_size * 2):
            state = np.random.normal(0, 1, self.state_dim)
            action = self.dqn_learner.select_action(state)
            next_state = np.random.normal(0, 1, self.state_dim)
            reward = np.random.normal(0, 1)
            done = bool(np.random.randint(2))
            
            self.dqn_learner.store_experience(state, action, reward, next_state, done)
            
        # Perform training step
        metrics = self.dqn_learner.train()
        
        # Check metrics
        self.assertIn('loss', metrics)
        self.assertIn('epsilon', metrics)
        self.assertIn('avg_q_value', metrics)
        
        # Check epsilon decay
        initial_epsilon = self.dqn_learner.epsilon
        self.dqn_learner.train()
        self.assertLess(self.dqn_learner.epsilon, initial_epsilon)
        
    def test_policy_training(self):
        """Test policy gradient training step."""
        # Fill buffer with random experiences
        for _ in range(self.policy_learner.batch_size * 2):
            state = np.random.normal(0, 1, self.state_dim)
            action = self.policy_learner.select_action(state)
            next_state = np.random.normal(0, 1, self.state_dim)
            reward = np.random.normal(0, 1)
            done = bool(np.random.randint(2))
            
            self.policy_learner.store_experience(state, action, reward, next_state, done)
            
        # Perform training step
        metrics = self.policy_learner.train()
        
        # Check metrics
        self.assertIn('loss', metrics)
        self.assertIn('avg_reward', metrics)
        self.assertIn('policy_std', metrics)
        
    def test_model_saving_loading(self):
        """Test model saving and loading."""
        import tempfile
        import os
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            model_path = tmp.name
            
            # Save model
            self.dqn_learner.save_model(model_path)
            
            # Create new learner and load model
            new_learner = ReinforcementLearner(
                self.state_dim,
                self.action_dim,
                learning_type='dqn'
            )
            new_learner.load_model(model_path)
            
            # Compare model parameters
            for p1, p2 in zip(self.dqn_learner.policy_net.parameters(),
                            new_learner.policy_net.parameters()):
                self.assertTrue(torch.equal(p1, p2))
                
            # Clean up
            os.unlink(model_path)
            
    def test_training_stats(self):
        """Test training statistics collection."""
        # Perform some training steps
        for _ in range(5):
            self.dqn_learner.training_stats.append({
                'loss': np.random.random(),
                'epsilon': self.dqn_learner.epsilon,
                'avg_q_value': np.random.random()
            })
            
        # Get stats DataFrame
        stats_df = self.dqn_learner.get_training_stats()
        
        # Check DataFrame structure
        self.assertEqual(len(stats_df), 5)
        self.assertIn('loss', stats_df.columns)
        self.assertIn('epsilon', stats_df.columns)
        self.assertIn('avg_q_value', stats_df.columns)
        
    def test_visualization(self):
        """Test visualization functionality."""
        # Add some training stats
        for _ in range(100):
            self.dqn_learner.training_stats.append({
                'loss': np.random.random(),
                'epsilon': self.dqn_learner.epsilon,
                'avg_q_value': np.random.random()
            })
            self.dqn_learner.episode_rewards.append(np.random.random())
            
        # Test plotting without saving
        self.dqn_learner.plot_training_progress()
        
        # Test plotting with saving
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            self.dqn_learner.plot_training_progress(save_path=tmp.name)

if __name__ == '__main__':
    unittest.main()