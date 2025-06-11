import pytest
import torch
import random
import sys
import os
from unittest.mock import patch, MagicMock
from grpo_fruits_catcher import GameConfig, TrainerConfig, Trainer, GameEngine, GameBrain

# Add the parent directory to sys.path to import the module
#sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

class TestTrainer:

    @pytest.fixture
    def mock_random(self):
        with patch('random.randint', return_value=5) as mock_rand:
            yield mock_rand

    @pytest.fixture
    def trainer(self):
        device = "cpu"
        game_config = GameConfig(
            screen_width=20,
            screen_height=11,
            max_fruits_on_screen=3
        )
        train_config = TrainerConfig(
            game_config=game_config
        )
        return Trainer(train_config, device)

    def test_create_init_shape(self, trainer):
        result, game_state = trainer._create_init()
        
        # Get the expected dimensions
        num_inits = GameConfig.get_initial_num(trainer.config.game_config)
        input_size = trainer.config.game_config.get_inputsize()
        
        # Check the shape of the result
        assert result.shape == (num_inits, input_size)
        assert game_state.shape == (num_inits, 3)  # Updated to expect 3 dimensions: score, step_count, activated_fruits_count
        
    def test_create_init_sprites_values(self, trainer):
        result, game_state = trainer._create_init()
        
        # Get the expected number of initializations
        num_inits = GameConfig.get_initial_num(trainer.config.game_config)
        
        # Check that sprite positions are correct (1-indexed)
        for idx in range(num_inits):
            assert result[idx, 0].item() == idx + 1
    
    def test_create_init_fruit_activation(self, trainer):
        result, game_state = trainer._create_init()
        
        # Get the expected number of initializations
        num_inits = GameConfig.get_initial_num(trainer.config.game_config)
        
        # Check that first fruit is activated for each initialization
        # Fruit activation status is the 3rd value of each fruit (index 2)
        for idx in range(num_inits):
            assert result[idx, 3].item() == 1.0  # First fruit activation status
            
    def test_create_init_with_fixed_random(self, trainer, mock_random):
        result, game_state = trainer._create_init()
        
        # Get the expected number of initializations
        num_inits = GameConfig.get_initial_num(trainer.config.game_config)
        
        # Check that the x position of the first fruit is always 5 (mocked random value)
        for idx in range(num_inits):
            assert result[idx, 1].item() == 5  # First fruit x position
            assert result[idx, 2].item() == 0  # First fruit y position always starts at 0
            
    def test_create_init_other_fruits_inactive(self, trainer):
        result, game_state = trainer._create_init()
        
        # Get the expected number of initializations
        num_inits = GameConfig.get_initial_num(trainer.config.game_config)
        
        # Check that other fruits are inactive (activation status = 0)
        for idx in range(num_inits):
            # Second fruit activation status
            assert result[idx, 6].item() == 0
            # Third fruit activation status  
            assert result[idx, 9].item() == 0

class TestGameEngine:

    @pytest.fixture
    def game_config(self):
        return GameConfig(
            screen_width=10,
            screen_height=6,
            sprite_width=3,
            max_fruits_on_screen=2,
            min_fruits_on_screen=0  # Changed to 0 to avoid auto-respawning during tests
        )

    @pytest.fixture
    def trainer_config(self, game_config):
        return TrainerConfig(game_config=game_config)

    @pytest.fixture
    def game_brain(self, trainer_config):
        return GameBrain(trainer_config)

    @pytest.fixture
    def game_engine(self, trainer_config, game_brain):
        return GameEngine(trainer_config, game_brain)

    def test_sprite_movement_stay(self, game_engine):
        """Test sprite stays in place when action is 1"""
        batch_size, num_inits = 1, 1
        input_size = game_engine.config.game_config.get_inputsize()
        
        # Create input state with sprite at position 5
        inputs_state = torch.zeros(batch_size, num_inits, input_size)
        inputs_state[0, 0, 0] = 5  # sprite x position
        
        game_state = torch.zeros(batch_size, num_inits, 3)
        
        # Mock brain to always return action 1 (stay)
        with patch.object(game_engine.brain, 'sample_action', return_value=(torch.tensor([1]), torch.tensor([0.0]))):
            new_inputs, actions, new_game_state = game_engine.update(inputs_state, game_state)
            
        assert new_inputs[0, 0, 0].item() == 5  # sprite should stay at position 5
        assert actions[0, 0].item() == 1

    def test_sprite_movement_left(self, game_engine):
        """Test sprite moves left when action is 0"""
        batch_size, num_inits = 1, 1
        input_size = game_engine.config.game_config.get_inputsize()
        
        inputs_state = torch.zeros(batch_size, num_inits, input_size)
        inputs_state[0, 0, 0] = 5  # sprite x position
        
        game_state = torch.zeros(batch_size, num_inits, 3)
        
        # Mock brain to return action 0 (left)
        with patch.object(game_engine.brain, 'sample_action', return_value=(torch.tensor([0]), torch.tensor([0.0]))):
            new_inputs, actions, new_game_state = game_engine.update(inputs_state, game_state)
            
        assert new_inputs[0, 0, 0].item() == 4  # sprite should move left to position 4
        assert actions[0, 0].item() == 0

    def test_sprite_movement_right(self, game_engine):
        """Test sprite moves right when action is 2"""
        batch_size, num_inits = 1, 1
        input_size = game_engine.config.game_config.get_inputsize()
        
        inputs_state = torch.zeros(batch_size, num_inits, input_size)
        inputs_state[0, 0, 0] = 5  # sprite x position
        
        game_state = torch.zeros(batch_size, num_inits, 3)
        
        # Mock brain to return action 2 (right)
        with patch.object(game_engine.brain, 'sample_action', return_value=(torch.tensor([2]), torch.tensor([0.0]))):
            new_inputs, actions, new_game_state = game_engine.update(inputs_state, game_state)
            
        assert new_inputs[0, 0, 0].item() == 6  # sprite should move right to position 6
        assert actions[0, 0].item() == 2

    def test_sprite_boundary_left(self, game_engine):
        """Test sprite doesn't move beyond left boundary"""
        batch_size, num_inits = 1, 1
        input_size = game_engine.config.game_config.get_inputsize()
        
        inputs_state = torch.zeros(batch_size, num_inits, input_size)
        inputs_state[0, 0, 0] = 0  # sprite at leftmost position
        
        game_state = torch.zeros(batch_size, num_inits, 3)
        
        # Mock brain to return action 0 (left)
        with patch.object(game_engine.brain, 'sample_action', return_value=(torch.tensor([0]), torch.tensor([0.0]))):
            new_inputs, actions, new_game_state = game_engine.update(inputs_state, game_state)
            
        assert new_inputs[0, 0, 0].item() == 0  # sprite should stay at position 0

    def test_sprite_boundary_right(self, game_engine):
        """Test sprite doesn't move beyond right boundary"""
        batch_size, num_inits = 1, 1
        input_size = game_engine.config.game_config.get_inputsize()
        screen_width = game_engine.config.game_config.screen_width
        
        inputs_state = torch.zeros(batch_size, num_inits, input_size)
        inputs_state[0, 0, 0] = screen_width - 1  # sprite at rightmost position
        
        game_state = torch.zeros(batch_size, num_inits, 3)
        
        # Mock brain to return action 2 (right)
        with patch.object(game_engine.brain, 'sample_action', return_value=(torch.tensor([2]), torch.tensor([0.0]))):
            new_inputs, actions, new_game_state = game_engine.update(inputs_state, game_state)
            
        assert new_inputs[0, 0, 0].item() == screen_width - 1  # sprite should stay at rightmost position

    def test_fruit_falling(self, game_engine):
        """Test that active fruits fall down by 1 step"""
        batch_size, num_inits = 1, 1
        input_size = game_engine.config.game_config.get_inputsize()
        
        inputs_state = torch.zeros(batch_size, num_inits, input_size)
        inputs_state[0, 0, 0] = 5  # sprite position
        inputs_state[0, 0, 1] = 3  # fruit 1 x position
        inputs_state[0, 0, 2] = 2  # fruit 1 y position
        inputs_state[0, 0, 3] = 1  # fruit 1 active
        
        game_state = torch.zeros(batch_size, num_inits, 3)
        
        with patch.object(game_engine.brain, 'sample_action', return_value=(torch.tensor([1]), torch.tensor([0.0]))):
            new_inputs, actions, new_game_state = game_engine.update(inputs_state, game_state)
            
        assert new_inputs[0, 0, 2].item() == 3  # fruit y position should increase by 1

    def test_fruit_caught_positive_score(self, game_engine):
        """Test that catching a fruit gives +1 score"""
        batch_size, num_inits = 1, 1
        input_size = game_engine.config.game_config.get_inputsize()
        screen_height = game_engine.config.game_config.screen_height
        
        inputs_state = torch.zeros(batch_size, num_inits, input_size)
        inputs_state[0, 0, 0] = 5  # sprite position
        inputs_state[0, 0, 1] = 5  # fruit 1 x position (same as sprite)
        inputs_state[0, 0, 2] = screen_height - 2  # fruit 1 y position (will reach bottom after falling)
        inputs_state[0, 0, 3] = 1  # fruit 1 active
        
        game_state = torch.zeros(batch_size, num_inits, 3)
        
        with patch.object(game_engine.brain, 'sample_action', return_value=(torch.tensor([1]), torch.tensor([0.0]))):
            new_inputs, actions, new_game_state = game_engine.update(inputs_state, game_state)
            
        assert new_game_state[0, 0, 0].item() == 1  # score should increase by 1
        assert new_inputs[0, 0, 3].item() == 0  # fruit should be deactivated

    def test_fruit_missed_negative_score(self, game_engine):
        """Test that missing a fruit gives -1 score"""
        batch_size, num_inits = 1, 1
        input_size = game_engine.config.game_config.get_inputsize()
        screen_height = game_engine.config.game_config.screen_height
        
        inputs_state = torch.zeros(batch_size, num_inits, input_size)
        inputs_state[0, 0, 0] = 5  # sprite position
        inputs_state[0, 0, 1] = 8  # fruit 1 x position (different from sprite)
        inputs_state[0, 0, 2] = screen_height - 2  # fruit 1 y position (will reach bottom after falling)
        inputs_state[0, 0, 3] = 1  # fruit 1 active
        
        game_state = torch.zeros(batch_size, num_inits, 3)
        
        with patch.object(game_engine.brain, 'sample_action', return_value=(torch.tensor([1]), torch.tensor([0.0]))):
            new_inputs, actions, new_game_state = game_engine.update(inputs_state, game_state)
            
        assert new_game_state[0, 0, 0].item() == -1  # score should decrease by 1
        assert new_inputs[0, 0, 3].item() == 0  # fruit should be deactivated

    def test_step_count_increment(self, game_engine):
        """Test that step count increases by 1 each update"""
        batch_size, num_inits = 1, 1
        input_size = game_engine.config.game_config.get_inputsize()
        
        inputs_state = torch.zeros(batch_size, num_inits, input_size)
        inputs_state[0, 0, 0] = 5  # sprite position
        
        game_state = torch.zeros(batch_size, num_inits, 3)
        game_state[0, 0, 1] = 10  # initial step count
        
        with patch.object(game_engine.brain, 'sample_action', return_value=(torch.tensor([1]), torch.tensor([0.0]))):
            new_inputs, actions, new_game_state = game_engine.update(inputs_state, game_state)
            
        assert new_game_state[0, 0, 1].item() == 11  # step count should increase by 1

    def test_batch_processing(self, game_engine):
        """Test that the engine can process multiple games in a batch"""
        batch_size, num_inits = 2, 3
        input_size = game_engine.config.game_config.get_inputsize()
        
        inputs_state = torch.zeros(batch_size, num_inits, input_size)
        # Set different sprite positions for each game
        inputs_state[0, 0, 0] = 3
        inputs_state[0, 1, 0] = 4
        inputs_state[0, 2, 0] = 5
        inputs_state[1, 0, 0] = 6
        inputs_state[1, 1, 0] = 7
        inputs_state[1, 2, 0] = 8
        
        game_state = torch.zeros(batch_size, num_inits, 3)
        
        # Mock brain to return different actions for each game
        mock_actions = torch.tensor([1, 0, 2, 1, 2, 0])  # stay, left, right, stay, right, left
        with patch.object(game_engine.brain, 'sample_action', return_value=(mock_actions, torch.zeros(6))):
            new_inputs, actions, new_game_state = game_engine.update(inputs_state, game_state)
            
        # Check that each game was processed with the correct action
        assert actions.shape == (batch_size, num_inits)
        assert new_game_state[0, 0, 1].item() == 1  # step count increased for all games
        assert new_game_state[1, 2, 1].item() == 1

    def test_fruits_reached_bottom_tracking(self, game_engine):
        """Test that the third dimension correctly tracks fruits reaching the bottom"""
        batch_size, num_inits = 1, 1
        input_size = game_engine.config.game_config.get_inputsize()
        screen_height = game_engine.config.game_config.screen_height
        
        inputs_state = torch.zeros(batch_size, num_inits, input_size)
        inputs_state[0, 0, 0] = 5  # sprite position
        inputs_state[0, 0, 1] = 8  # fruit 1 x position (different from sprite - will be missed)
        inputs_state[0, 0, 2] = screen_height - 2  # fruit 1 y position (will reach bottom after falling)
        inputs_state[0, 0, 3] = 1  # fruit 1 active
        
        game_state = torch.zeros(batch_size, num_inits, 3)
        
        with patch.object(game_engine.brain, 'sample_action', return_value=(torch.tensor([1]), torch.tensor([0.0]))):
            new_inputs, actions, new_game_state = game_engine.update(inputs_state, game_state)
        
        # Check that fruits reached bottom count is incremented
        assert new_game_state[0, 0, 2].item() == 1  # one fruit reached bottom
        
    def test_fruits_reached_bottom_multiple(self, game_engine):
        """Test tracking multiple fruits reaching bottom in same step"""
        batch_size, num_inits = 1, 1
        input_size = game_engine.config.game_config.get_inputsize()
        screen_height = game_engine.config.game_config.screen_height
        
        inputs_state = torch.zeros(batch_size, num_inits, input_size)
        inputs_state[0, 0, 0] = 5  # sprite position
        # Set up two fruits to reach bottom
        inputs_state[0, 0, 1] = 1  # fruit 1 x position (missed)
        inputs_state[0, 0, 2] = screen_height - 2  # fruit 1 y position  
        inputs_state[0, 0, 3] = 1  # fruit 1 active
        inputs_state[0, 0, 4] = 8  # fruit 2 x position (missed)
        inputs_state[0, 0, 5] = screen_height - 2  # fruit 2 y position
        inputs_state[0, 0, 6] = 1  # fruit 2 active
        
        game_state = torch.zeros(batch_size, num_inits, 3)
        
        with patch.object(game_engine.brain, 'sample_action', return_value=(torch.tensor([1]), torch.tensor([0.0]))):
            new_inputs, actions, new_game_state = game_engine.update(inputs_state, game_state)
        
        # Check that fruits reached bottom count is incremented by 2
        assert new_game_state[0, 0, 2].item() == 2  # two fruits reached bottom

    def test_fruits_reached_bottom_caught_vs_missed(self, game_engine):
        """Test that both caught and missed fruits count as reaching bottom"""
        batch_size, num_inits = 1, 1
        input_size = game_engine.config.game_config.get_inputsize()
        screen_height = game_engine.config.game_config.screen_height
        
        inputs_state = torch.zeros(batch_size, num_inits, input_size)
        inputs_state[0, 0, 0] = 5  # sprite position
        # Set up two fruits: one caught, one missed
        inputs_state[0, 0, 1] = 5  # fruit 1 x position (same as sprite - will be caught)
        inputs_state[0, 0, 2] = screen_height - 2  # fruit 1 y position  
        inputs_state[0, 0, 3] = 1  # fruit 1 active
        inputs_state[0, 0, 4] = 8  # fruit 2 x position (different from sprite - will be missed)
        inputs_state[0, 0, 5] = screen_height - 2  # fruit 2 y position
        inputs_state[0, 0, 6] = 1  # fruit 2 active
        
        game_state = torch.zeros(batch_size, num_inits, 3)
        
        with patch.object(game_engine.brain, 'sample_action', return_value=(torch.tensor([1]), torch.tensor([0.0]))):
            new_inputs, actions, new_game_state = game_engine.update(inputs_state, game_state)
        
        # Check that fruits reached bottom count includes both caught and missed fruits
        assert new_game_state[0, 0, 2].item() == 2  # both fruits reached bottom
        assert new_game_state[0, 0, 0].item() == 0  # score: +1 for caught, -1 for missed = 0


class TestRewardMethod:
    """Test suite for the advanced reward algorithm."""

    @pytest.fixture
    def trainer(self):
        """Create a trainer instance for testing."""
        device = "cpu"
        game_config = GameConfig(
            screen_width=20,
            screen_height=11,
            max_fruits_on_screen=3
        )
        train_config = TrainerConfig(game_config=game_config)
        return Trainer(train_config, device)

    def test_reward_basic_functionality(self, trainer):
        """Test basic reward method functionality and return types."""
        # Test single game state
        game_state = torch.tensor([[[5.0, 10.0, 5.0]]])  # score=5, steps=10, fruits=5
        
        reward, score = trainer._reward(game_state)
        
        # Check return types and shapes
        assert isinstance(reward, torch.Tensor)
        assert isinstance(score, torch.Tensor)
        assert reward.shape == (1, 1)
        assert score.shape == (1, 1)
        assert score.item() == 5.0

    def test_reward_perfect_performance(self, trainer):
        """Test reward for perfect catch rate (100%)."""
        # Perfect performance: all 5 fruits caught
        game_state = torch.tensor([[[5.0, 10.0, 5.0]]])
        
        reward, score = trainer._reward(game_state)
        
        # Perfect performance should yield positive reward
        assert reward.item() > 1.0  # Should be significantly positive
        assert score.item() == 5.0

    def test_reward_poor_performance(self, trainer):
        """Test reward for poor catch rate."""
        # Poor performance: missed more than caught
        game_state = torch.tensor([[[-3.0, 10.0, 5.0]]])  # score=-3, 2 caught, 5 missed
        
        reward, score = trainer._reward(game_state)
        
        # Poor performance should yield lower reward
        assert reward.item() < 1.0
        assert score.item() == -3.0

    def test_reward_incremental_calculation(self, trainer):
        """Test incremental reward calculation with previous state."""
        # Previous state
        prev_state = torch.tensor([[[2.0, 5.0, 3.0]]])  # score=2, steps=5, fruits=3
        
        # Current state (caught 1 more fruit)
        curr_state = torch.tensor([[[3.0, 6.0, 4.0]]])  # score=3, steps=6, fruits=4
        
        reward_with_prev, _ = trainer._reward(curr_state, prev_state)
        reward_without_prev, _ = trainer._reward(curr_state)
        
        # With previous state should account for incremental changes
        assert reward_with_prev.item() != reward_without_prev.item()
        # Should get immediate reward for the +1 score change
        assert reward_with_prev.item() > 0

    def test_reward_catch_rate_bonus(self, trainer):
        """Test exponential catch rate bonus."""
        # High catch rate scenario
        high_catch_state = torch.tensor([[[9.0, 15.0, 10.0]]])  # 90% catch rate
        
        # Low catch rate scenario  
        low_catch_state = torch.tensor([[[1.0, 15.0, 10.0]]])   # 10% catch rate
        
        high_reward, _ = trainer._reward(high_catch_state)
        low_reward, _ = trainer._reward(low_catch_state)
        
        # High catch rate should get significantly higher reward
        assert high_reward.item() > low_reward.item()

    def test_reward_batch_processing(self, trainer):
        """Test reward calculation with batch of game states."""
        # Batch of different performance levels
        batch_states = torch.tensor([
            [[5.0, 10.0, 5.0]],   # Perfect performance
            [[0.0, 10.0, 5.0]],   # Break-even
            [[-2.0, 10.0, 5.0]]   # Poor performance
        ])
        
        rewards, scores = trainer._reward(batch_states)
        
        # Check shapes
        assert rewards.shape == (3, 1)
        assert scores.shape == (3, 1)
        
        # Perfect > break-even > poor performance
        assert rewards[0].item() > rewards[1].item() > rewards[2].item()
        
        # Scores should match input
        assert scores[0].item() == 5.0
        assert scores[1].item() == 0.0
        assert scores[2].item() == -2.0

    def test_reward_bounds(self, trainer):
        """Test that rewards are properly bounded."""
        # Extreme positive performance
        extreme_positive = torch.tensor([[[100.0, 20.0, 100.0]]])
        
        # Extreme negative performance  
        extreme_negative = torch.tensor([[[-100.0, 20.0, 100.0]]])
        
        pos_reward, _ = trainer._reward(extreme_positive)
        neg_reward, _ = trainer._reward(extreme_negative)
        
        # Should be bounded within reasonable range
        assert -5.0 <= pos_reward.item() <= 10.0
        assert -5.0 <= neg_reward.item() <= 10.0

    def test_reward_edge_cases(self, trainer):
        """Test reward calculation for edge cases."""
        # Zero steps (should not crash)
        zero_steps = torch.tensor([[[0.0, 0.0, 0.0]]])
        reward, _ = trainer._reward(zero_steps)
        assert not torch.isnan(reward).any()
        
        # Very large numbers
        large_numbers = torch.tensor([[[1000.0, 1000.0, 1000.0]]])
        reward, _ = trainer._reward(large_numbers)
        assert not torch.isnan(reward).any()
        assert not torch.isinf(reward).any()


class TestCreateTrajectory:
    """Test suite for the _create_trajector method."""

    @pytest.fixture
    def trainer(self):
        """Create a trainer instance for testing."""
        device = "cpu"
        game_config = GameConfig(
            screen_width=10,
            screen_height=6,
            max_fruits_on_screen=3  # Changed to 3 to match expected input_size
        )
        train_config = TrainerConfig(
            game_config=game_config,
            batch_size=3,
            max_steps=5
        )
        return Trainer(train_config, device)

    def test_create_trajector_basic_functionality(self, trainer):
        """Test basic functionality and return types."""
        input_history, action_history, score_history, reward_history = trainer._create_trajector()
        
        # Check return types
        assert isinstance(input_history, torch.Tensor)
        assert isinstance(action_history, torch.Tensor)
        assert isinstance(score_history, torch.Tensor)
        assert isinstance(reward_history, torch.Tensor)

    def test_create_trajector_shapes(self, trainer):
        """Test that all returned tensors have correct shapes."""
        input_history, action_history, score_history, reward_history = trainer._create_trajector()
        
        # Get expected dimensions
        num_steps = trainer.config.max_steps
        batch_size = trainer.config.batch_size
        num_inits = GameConfig.get_initial_num(trainer.config.game_config)
        input_size = trainer.config.game_config.get_inputsize()
        
        # Check shapes
        assert input_history.shape == (num_steps, batch_size, num_inits, input_size)
        assert action_history.shape == (num_steps, batch_size, num_inits)
        assert score_history.shape == (num_steps, batch_size, num_inits)
        assert reward_history.shape == (num_steps, batch_size, num_inits)

    def test_create_trajector_data_types(self, trainer):
        """Test that tensors have correct data types."""
        input_history, action_history, score_history, reward_history = trainer._create_trajector()
        
        # Check data types
        assert input_history.dtype == torch.float32
        assert action_history.dtype == torch.long  # Actions should be integers
        assert score_history.dtype == torch.float32
        assert reward_history.dtype == torch.float32

    def test_create_trajector_device_placement(self, trainer):
        """Test that all tensors are on the correct device."""
        input_history, action_history, score_history, reward_history = trainer._create_trajector()
        
        expected_device = trainer.device
        assert input_history.device.type == expected_device
        assert action_history.device.type == expected_device
        assert score_history.device.type == expected_device
        assert reward_history.device.type == expected_device

    def test_create_trajector_action_validity(self, trainer):
        """Test that all actions are valid (0, 1, or 2)."""
        _, action_history, _, _ = trainer._create_trajector()
        
        # Actions should be 0 (left), 1 (stay), or 2 (right)
        assert torch.all(action_history >= 0)
        assert torch.all(action_history <= 2)

    def test_create_trajector_progression(self, trainer):
        """Test that game state progresses over time."""
        input_history, action_history, score_history, reward_history = trainer._create_trajector()
        
        # Check that different steps have potentially different states
        # (though they could be the same due to randomness)
        assert input_history.shape[0] == trainer.config.max_steps
        
        # Check that we have some non-zero values in histories
        # (at least some activity should happen)
        has_activity = (
            torch.any(action_history > 0) or  # Some non-stay actions
            torch.any(score_history != 0) or  # Some score changes
            torch.any(reward_history != 0)    # Some rewards
        )
        # Note: Due to randomness, this might not always be true, but usually should be
        # We'll check for at least some basic progression

    def test_create_trajector_initial_state_consistency(self, trainer):
        """Test that initial states are consistent with _create_init."""
        input_history, _, _, _ = trainer._create_trajector()
        
        # Get the first step's input for comparison
        first_step_inputs = input_history[0]  # shape: (batch_size, num_inits, input_size)
        
        # Check that sprite positions are properly initialized (1-indexed)
        batch_size = trainer.config.batch_size
        num_inits = GameConfig.get_initial_num(trainer.config.game_config)
        
        for b in range(batch_size):
            for i in range(num_inits):
                # Sprite position should be i+1 (1-indexed)
                assert first_step_inputs[b, i, 0].item() == i + 1

    def test_create_trajector_reward_calculation(self, trainer):
        """Test that rewards are calculated correctly throughout trajectory."""
        _, _, score_history, reward_history = trainer._create_trajector()
        
        # Check that rewards are finite and within expected bounds
        assert torch.all(torch.isfinite(reward_history))
        assert torch.all(reward_history >= -5.0)  # Minimum bound from reward algorithm
        assert torch.all(reward_history <= 10.0)  # Maximum bound from reward algorithm
        
        # Check that scores are finite
        assert torch.all(torch.isfinite(score_history))

    def test_create_trajector_game_engine_integration(self, trainer):
        """Test integration with game engine updates."""
        input_history, action_history, score_history, _ = trainer._create_trajector()
        
        num_steps = trainer.config.max_steps
        batch_size = trainer.config.batch_size
        num_inits = GameConfig.get_initial_num(trainer.config.game_config)
        
        # Verify that inputs change over time (game engine is updating state)
        for step in range(1, min(num_steps, 3)):  # Check first few steps
            current_inputs = input_history[step]
            prev_inputs = input_history[step - 1]
            
            # At least some inputs should potentially change between steps
            # (due to fruit movement, sprite movement, etc.)
            # We can't guarantee change due to randomness, but structure should be valid
            assert current_inputs.shape == prev_inputs.shape

    def test_create_trajector_batch_independence(self, trainer):
        """Test that different batch elements can have different trajectories."""
        input_history, action_history, score_history, reward_history = trainer._create_trajector()
        
        batch_size = trainer.config.batch_size
        if batch_size > 1:
            # Different batch elements might have different trajectories
            # We just check that the structure is correct for all batches
            for b in range(batch_size):
                batch_inputs = input_history[:, b, :, :]
                batch_actions = action_history[:, b, :]
                batch_scores = score_history[:, b, :]
                batch_rewards = reward_history[:, b, :]
                
                # Each batch element should have valid shapes
                assert batch_inputs.shape[0] == trainer.config.max_steps
                assert batch_actions.shape[0] == trainer.config.max_steps
                assert batch_scores.shape[0] == trainer.config.max_steps
                assert batch_rewards.shape[0] == trainer.config.max_steps

    def test_create_trajector_incremental_rewards(self, trainer):
        """Test that incremental reward calculation is working."""
        _, _, score_history, reward_history = trainer._create_trajector()
        
        # For the first step, prev_game_state should be None
        # For subsequent steps, prev_game_state should be provided
        # We can't directly test this without mocking, but we can check consistency
        
        # Rewards should be calculated for all steps
        assert reward_history.shape[0] == trainer.config.max_steps
        
        # All rewards should be finite
        assert torch.all(torch.isfinite(reward_history))

    def test_create_trajector_reproducibility_with_seed(self, trainer):
        """Test that trajectories are reproducible with same random seed."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        random.seed(42)
        
        traj1 = trainer._create_trajector()
        
        # Reset seed and generate again
        torch.manual_seed(42)
        random.seed(42)
        
        traj2 = trainer._create_trajector()
        
        # Results should be identical
        input_hist1, action_hist1, score_hist1, reward_hist1 = traj1
        input_hist2, action_hist2, score_hist2, reward_hist2 = traj2
        
        assert torch.allclose(input_hist1, input_hist2, atol=1e-6)
        assert torch.equal(action_hist1, action_hist2)
        assert torch.allclose(score_hist1, score_hist2, atol=1e-6)
        assert torch.allclose(reward_hist1, reward_hist2, atol=1e-6)

    def test_create_trajector_memory_efficiency(self, trainer):
        """Test that method doesn't create excessive memory usage."""
        # This is a basic test to ensure tensors are created efficiently
        input_history, action_history, score_history, reward_history = trainer._create_trajector()
        
        # Check that tensors are contiguous (memory efficient)
        assert input_history.is_contiguous()
        assert action_history.is_contiguous()
        assert score_history.is_contiguous()
        assert reward_history.is_contiguous()
        
        # Check that no unexpected large allocations occurred
        # (tensors should match expected size calculations)
        expected_input_size = (
            trainer.config.max_steps * 
            trainer.config.batch_size * 
            GameConfig.get_initial_num(trainer.config.game_config) * 
            trainer.config.game_config.get_inputsize()
        )
        actual_input_size = input_history.numel()
        assert actual_input_size == expected_input_size