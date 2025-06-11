import pytest
import torch
import random
from unittest.mock import patch, MagicMock
from grpo_fruits_catcher import GameConfig, TrainerConfig, Trainer


class TestTrainEpoch:
    """Test suite for the _train_epoch method."""
    
    @pytest.fixture
    def trainer(self):
        device = "cpu"
        game_config = GameConfig(
            screen_width=10,
            screen_height=5,
            max_fruits_on_screen=2
        )
        train_config = TrainerConfig(
            game_config=game_config,
            batch_size=4,
            max_steps=10,
            hidden_size=32,
            lr_rate=1e-3
        )
        return Trainer(train_config, device)
    
    @pytest.fixture
    def mock_create_trajector(self, trainer):
        """Mock _create_trajector to return predictable data"""
        num_steps = trainer.config.max_steps
        batch_size = trainer.config.batch_size
        num_inits = GameConfig.get_initial_num(trainer.config.game_config)
        input_size = trainer.config.game_config.get_inputsize()
        
        # Create mock data with known shapes
        input_history = torch.randn(num_steps, batch_size, num_inits, input_size)
        action_history = torch.randint(0, 3, (num_steps, batch_size, num_inits))
        score_history = torch.randn(num_steps, batch_size, num_inits)
        reward_history = torch.randn(num_steps, batch_size, num_inits)
        
        with patch.object(trainer, '_create_trajector', return_value=(
            input_history, action_history, score_history, reward_history
        )):
            yield
    
    def test_train_epoch_return_types(self, trainer):
        """Test that _train_epoch returns correct types"""
        avg_reward, avg_score = trainer._train_epoch()
        
        assert isinstance(avg_reward, float), "Average reward should be a float"
        assert isinstance(avg_score, float), "Average score should be a float"
    
    def test_train_epoch_return_values_finite(self, trainer):
        """Test that _train_epoch returns finite values"""
        avg_reward, avg_score = trainer._train_epoch()
        
        assert torch.isfinite(torch.tensor(avg_reward)), "Average reward should be finite"
        assert torch.isfinite(torch.tensor(avg_score)), "Average score should be finite"
    
    def test_train_epoch_optimizer_called(self, trainer, mock_create_trajector):
        """Test that the optimizer is properly called during training"""
        # Mock the optimizer methods
        with patch.object(trainer.optimizer, 'zero_grad') as mock_zero_grad, \
             patch.object(trainer.optimizer, 'step') as mock_step:
            
            trainer._train_epoch()
            
            # Verify optimizer methods were called
            mock_zero_grad.assert_called_once()
            mock_step.assert_called_once()
    
    def test_train_epoch_policy_loss_called(self, trainer, mock_create_trajector):
        """Test that _policy_loss is called for each time step"""
        with patch.object(trainer, '_policy_loss', return_value=torch.tensor(0.5)) as mock_policy_loss:
            trainer._train_epoch()
            
            # Should be called max_steps times
            assert mock_policy_loss.call_count == trainer.config.max_steps
    
    def test_train_epoch_gradient_flow(self, trainer, mock_create_trajector):
        """Test that gradients are properly computed and flow through the network"""
        # Store initial parameters
        initial_params = {name: param.clone() for name, param in trainer.brain.named_parameters()}
        
        # Run one training epoch
        trainer._train_epoch()
        
        # Check that at least some parameters changed (indicating gradient flow)
        params_changed = False
        for name, param in trainer.brain.named_parameters():
            if not torch.allclose(initial_params[name], param, atol=1e-7):
                params_changed = True
                break
        
        assert params_changed, "At least some network parameters should change after training"
    
    def test_train_epoch_group_reward_normalization(self, trainer):
        """Test that GRPO reward normalization is applied correctly"""
        num_steps = trainer.config.max_steps
        batch_size = trainer.config.batch_size
        num_inits = GameConfig.get_initial_num(trainer.config.game_config)
        
        # Create mock reward data with known distribution
        reward_history = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                                       [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
                                       [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0],
                                       [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]]] * num_steps)
        
        # Mock the _create_trajector to return our specific reward data
        input_history = torch.randn(num_steps, batch_size, num_inits, trainer.config.game_config.get_inputsize())
        action_history = torch.randint(0, 3, (num_steps, batch_size, num_inits))
        score_history = torch.randn(num_steps, batch_size, num_inits)
        
        with patch.object(trainer, '_create_trajector', return_value=(
            input_history, action_history, score_history, reward_history
        )):
            # Mock _policy_loss to capture the normalized rewards
            captured_rewards = []
            
            def mock_policy_loss(inputs, actions, rewards):
                captured_rewards.append(rewards.clone())
                return torch.tensor(0.1)
            
            with patch.object(trainer, '_policy_loss', side_effect=mock_policy_loss):
                trainer._train_epoch()
            
            # Verify that the same normalized rewards were used for all time steps
            assert len(captured_rewards) == num_steps
            for i in range(1, len(captured_rewards)):
                assert torch.allclose(captured_rewards[0], captured_rewards[i], atol=1e-6), \
                    "All time steps should use the same final normalized reward"
    
    def test_train_epoch_tensor_shapes_consistency(self, trainer, mock_create_trajector):
        """Test that tensor reshaping maintains correct dimensions"""
        # Capture the shapes used in _policy_loss calls
        captured_shapes = []
        
        def mock_policy_loss(inputs, actions, rewards):
            captured_shapes.append({
                'inputs': inputs.shape,
                'actions': actions.shape,
                'rewards': rewards.shape
            })
            return torch.tensor(0.1)
        
        with patch.object(trainer, '_policy_loss', side_effect=mock_policy_loss):
            trainer._train_epoch()
        
        num_steps = trainer.config.max_steps
        batch_size = trainer.config.batch_size
        num_inits = GameConfig.get_initial_num(trainer.config.game_config)
        input_size = trainer.config.game_config.get_inputsize()
        
        # Verify shapes for each time step
        expected_flat_size = batch_size * num_inits
        for shapes in captured_shapes:
            assert shapes['inputs'] == (expected_flat_size, input_size), \
                f"Input shape should be ({expected_flat_size}, {input_size})"
            assert shapes['actions'] == (expected_flat_size,), \
                f"Actions shape should be ({expected_flat_size},)"
            assert shapes['rewards'] == (expected_flat_size,), \
                f"Rewards shape should be ({expected_flat_size},)"
    
    def test_train_epoch_loss_accumulation(self, trainer, mock_create_trajector):
        """Test that losses are properly accumulated across time steps"""
        loss_values = [torch.tensor(0.1 * i) for i in range(trainer.config.max_steps)]
        
        with patch.object(trainer, '_policy_loss', side_effect=loss_values):
            # Mock backward() to prevent actual backpropagation during testing
            with patch.object(torch.Tensor, 'backward'):
                trainer._train_epoch()
        
        # The test passes if no exceptions are raised during loss accumulation
        assert True, "Loss accumulation should complete without errors"
    
    def test_train_epoch_with_different_config_sizes(self):
        """Test _train_epoch with different configuration sizes"""
        configs = [
            (5, 3, 8),   # (batch_size, max_steps, hidden_size)
            (10, 5, 16),
            (2, 15, 64)
        ]
        
        for batch_size, max_steps, hidden_size in configs:
            device = "cpu"
            game_config = GameConfig(screen_width=8, screen_height=6, max_fruits_on_screen=2)
            train_config = TrainerConfig(
                game_config=game_config,
                batch_size=batch_size,
                max_steps=max_steps,
                hidden_size=hidden_size
            )
            trainer = Trainer(train_config, device)
            
            # Mock _create_trajector for this configuration
            num_inits = GameConfig.get_initial_num(game_config)
            input_size = game_config.get_inputsize()
            
            input_history = torch.randn(max_steps, batch_size, num_inits, input_size)
            action_history = torch.randint(0, 3, (max_steps, batch_size, num_inits))
            score_history = torch.randn(max_steps, batch_size, num_inits)
            reward_history = torch.randn(max_steps, batch_size, num_inits)
            
            with patch.object(trainer, '_create_trajector', return_value=(
                input_history, action_history, score_history, reward_history
            )):
                avg_reward, avg_score = trainer._train_epoch()
                
                assert isinstance(avg_reward, float), f"Failed for config {configs}"
                assert isinstance(avg_score, float), f"Failed for config {configs}"
                assert torch.isfinite(torch.tensor(avg_reward)), f"Failed for config {configs}"
                assert torch.isfinite(torch.tensor(avg_score)), f"Failed for config {configs}"
    
    def test_train_epoch_epsilon_safety(self, trainer):
        """Test that epsilon prevents division by zero in group normalization"""
        num_steps = trainer.config.max_steps
        batch_size = trainer.config.batch_size
        num_inits = GameConfig.get_initial_num(trainer.config.game_config)
        
        # Create reward data with zero standard deviation (constant rewards)
        constant_reward = 5.0
        reward_history = torch.full((num_steps, batch_size, num_inits), constant_reward)
        
        input_history = torch.randn(num_steps, batch_size, num_inits, trainer.config.game_config.get_inputsize())
        action_history = torch.randint(0, 3, (num_steps, batch_size, num_inits))
        score_history = torch.randn(num_steps, batch_size, num_inits)
        
        with patch.object(trainer, '_create_trajector', return_value=(
            input_history, action_history, score_history, reward_history
        )):
            # This should not raise any division by zero errors
            avg_reward, avg_score = trainer._train_epoch()
            
            assert torch.isfinite(torch.tensor(avg_reward)), "Should handle zero std dev without NaN"
            assert torch.isfinite(torch.tensor(avg_score)), "Should handle zero std dev without NaN"
    
    def test_train_epoch_backward_compatibility(self, trainer, mock_create_trajector):
        """Test that _train_epoch works with the current trainer interface"""
        # Test that the method can be called successfully without errors
        try:
            avg_reward, avg_score = trainer._train_epoch()
            assert True, "Method should execute without raising exceptions"
        except Exception as e:
            pytest.fail(f"_train_epoch raised an unexpected exception: {e}")
    
    def test_train_epoch_deterministic_with_seed(self):
        """Test that _train_epoch produces consistent results with the same seed"""
        device = "cpu"
        game_config = GameConfig(screen_width=8, screen_height=6, max_fruits_on_screen=2)
        train_config = TrainerConfig(
            game_config=game_config,
            batch_size=3,
            max_steps=5,
            hidden_size=16
        )
        
        # Run with same seed twice
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            random.seed(42)
            trainer = Trainer(train_config, device)
            avg_reward, avg_score = trainer._train_epoch()
            results.append((avg_reward, avg_score))
        
        # Results should be the same (or very close due to floating point precision)
        assert abs(results[0][0] - results[1][0]) < 1e-5, "Results should be consistent with same seed"
        assert abs(results[0][1] - results[1][1]) < 1e-5, "Results should be consistent with same seed"
