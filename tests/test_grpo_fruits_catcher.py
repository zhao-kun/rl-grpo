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
        assert game_state.shape == (num_inits, 2)
        
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
        
        game_state = torch.zeros(batch_size, num_inits, 2)
        
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
        
        game_state = torch.zeros(batch_size, num_inits, 2)
        
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
        
        game_state = torch.zeros(batch_size, num_inits, 2)
        
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
        
        game_state = torch.zeros(batch_size, num_inits, 2)
        
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
        
        game_state = torch.zeros(batch_size, num_inits, 2)
        
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
        
        game_state = torch.zeros(batch_size, num_inits, 2)
        
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
        
        game_state = torch.zeros(batch_size, num_inits, 2)
        
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
        
        game_state = torch.zeros(batch_size, num_inits, 2)
        
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
        
        game_state = torch.zeros(batch_size, num_inits, 2)
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
        
        game_state = torch.zeros(batch_size, num_inits, 2)
        
        # Mock brain to return different actions for each game
        mock_actions = torch.tensor([1, 0, 2, 1, 2, 0])  # stay, left, right, stay, right, left
        with patch.object(game_engine.brain, 'sample_action', return_value=(mock_actions, torch.zeros(6))):
            new_inputs, actions, new_game_state = game_engine.update(inputs_state, game_state)
            
        # Check that each game was processed with the correct action
        assert actions.shape == (batch_size, num_inits)
        assert new_game_state[0, 0, 1].item() == 1  # step count increased for all games
        assert new_game_state[1, 2, 1].item() == 1


# Existing TestTrainer class continues here...