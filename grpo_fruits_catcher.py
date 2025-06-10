from dataclasses import dataclass, field
from typing import Optional, Tuple
from torch import nn

import torch
import torch.nn.functional as F 
import random

@dataclass
class GameConfig:
    """Configuration for the fruit catching game"""
    screen_width: int = 20
    screen_height: int = 11
    sprite_width: int = 3
    sprite_height: int = 1
    max_fruits_on_screen: int = 3
    min_fruits_on_screen: int = 1
    fruit_spawn_interval: int = 2  # frames between fruit spawns

    def get_inputsize(self):
        # Sprite position (1) + fruits data (max_fruits * 3 dimensions)
        return 1 + self.max_fruits_on_screen * 3
    
    @staticmethod
    def get_initial_num(config):
        return config.screen_width


@dataclass
class TrainerConfig:
    hidden_size: int = 128
    batch_size: int = 50
    total_epochs: int = 20
    max_steps: int = 200  # maximum number of steps before game ends
    game_config: Optional[GameConfig] = field(default_factory=GameConfig)


class GameBrain(nn.Module):
    
    def __init__(self, config: TrainerConfig):
        super(GameBrain, self).__init__()
        self.input_size = config.game_config.get_inputsize()
        self.fc1 = nn.Linear(self.input_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, 3)  # 3 actions: left, stay, right
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Action logits of shape (batch_size, output_size)
        """
        # Input -> Hidden layer with ReLU activation
        hidden = F.relu(self.fc1(x))
        
        # Hidden -> Output layer (logits)
        logits = self.fc2(hidden)
        
        return logits
    
    def get_action_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities using softmax"""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def sample_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Args:
            x: Input state tensor
            
        Returns:
            (action, log_probability)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        
        # Sample action from probability distribution
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob


class GameEngine:
    
    def __init__(self, config: TrainerConfig, brain: GameBrain ):
        self.config = config
        self.brain = brain

    def update(self, inputs_state: torch.Tensor, game_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update the game state for all games in the batch.
        
        Args:
            inputs_state: shape (batch_size, num_inits, input_size) - current game states
            game_state: shape (batch_size, num_inits, 3) - [score, step_count, activated_fruits_count] for each game
            
        Returns:
            new_inputs_state: updated game states
            actions: chosen actions for each game
            new_game_state: updated game states with scores, step counts, and activated fruits count
        """
        batch_size, num_inits, input_size = inputs_state.shape
        game_config = self.config.game_config
        
        # Get actions from brain for all games
        inputs_flat = inputs_state.view(-1, input_size)  # (batch_size * num_inits, input_size)
        actions, _ = self.brain.sample_action(inputs_flat)
        actions = actions.view(batch_size, num_inits)  # (batch_size, num_inits)
        
        # Create new states
        new_inputs_state = inputs_state.clone()
        new_game_state = game_state.clone()
        
        for b in range(batch_size):
            for i in range(num_inits):
                # Extract current sprite position (x coordinate)
                sprite_x = int(new_inputs_state[b, i, 0].item())
                
                # Move sprite based on action: 0=left, 1=stay, 2=right
                action = actions[b, i].item()
                if action == 0:  # left
                    sprite_x = max(0, sprite_x - 1)
                elif action == 2:  # right
                    sprite_x = min(game_config.screen_width - 1, sprite_x + 1)
                # action == 1 means stay (no change)
                
                new_inputs_state[b, i, 0] = sprite_x
                
                # Process fruits (3 fruits, each with 3 dimensions: x, y, activation)
                score_change = 0
                activated_fruits_in_step = 0
                fruits_deactivated_this_step = 0
                
                for fruit_idx in range(game_config.max_fruits_on_screen):
                    fruit_start = 1 + fruit_idx * 3
                    fruit_x = int(new_inputs_state[b, i, fruit_start].item())
                    fruit_y = int(new_inputs_state[b, i, fruit_start + 1].item())
                    fruit_active = int(new_inputs_state[b, i, fruit_start + 2].item())
                    
                    if fruit_active == 1:
                        # Move fruit down
                        fruit_y += 1
                        new_inputs_state[b, i, fruit_start + 1] = fruit_y
                        
                        # Check collision with sprite
                        sprite_y = game_config.screen_height - 1
                        if fruit_y == sprite_y:
                            # Check if sprite catches fruit
                            sprite_left = sprite_x - game_config.sprite_width // 2
                            sprite_right = sprite_x + game_config.sprite_width // 2
                            
                            if sprite_left <= fruit_x <= sprite_right:
                                # Sprite caught the fruit
                                score_change += 1
                                new_inputs_state[b, i, fruit_start + 2] = 0  # deactivate fruit
                                fruits_deactivated_this_step += 1
                            else:
                                # Fruit reached bottom without being caught
                                score_change -= 1
                                new_inputs_state[b, i, fruit_start + 2] = 0  # deactivate fruit
                                fruits_deactivated_this_step += 1
                        elif fruit_y >= game_config.screen_height:
                            # Fruit went off screen
                            new_inputs_state[b, i, fruit_start + 2] = 0  # deactivate fruit
                            fruits_deactivated_this_step += 1
                        else:
                            pass
                
                # Count currently active fruits after processing
                active_fruits = sum(new_inputs_state[b, i, 1 + j * 3 + 2].item() for j in range(game_config.max_fruits_on_screen))
                
                # Only spawn new fruit if no fruits were deactivated this step (to avoid immediate respawning)
                if (fruits_deactivated_this_step == 0 and
                    active_fruits < game_config.max_fruits_on_screen and 
                    active_fruits >= game_config.min_fruits_on_screen and 
                    random.randint(0, 1) == 1):
                    
                    # Find an inactive fruit slot
                    for fruit_idx in range(game_config.max_fruits_on_screen):
                        fruit_start = 1 + fruit_idx * 3
                        if new_inputs_state[b, i, fruit_start + 2].item() == 0:  # inactive
                            new_inputs_state[b, i, fruit_start] = random.randint(0, game_config.screen_width - 1)  # x
                            new_inputs_state[b, i, fruit_start + 1] = 0  # y
                            new_inputs_state[b, i, fruit_start + 2] = 1  # activate
                            activated_fruits_in_step += 1
                            break
                
                # Ensure minimum fruits are active
                current_active = sum(new_inputs_state[b, i, 1 + j * 3 + 2].item() for j in range(game_config.max_fruits_on_screen))
                while current_active < game_config.min_fruits_on_screen:
                    for fruit_idx in range(game_config.max_fruits_on_screen):
                        fruit_start = 1 + fruit_idx * 3
                        if new_inputs_state[b, i, fruit_start + 2].item() == 0:  # inactive
                            new_inputs_state[b, i, fruit_start] = random.randint(0, game_config.screen_width - 1)  # x
                            new_inputs_state[b, i, fruit_start + 1] = 0  # y
                            new_inputs_state[b, i, fruit_start + 2] = 1  # activate
                            activated_fruits_in_step += 1
                            current_active += 1
                            break
                    if current_active >= game_config.min_fruits_on_screen:
                        break
                
                # Update score, step count, and activated fruits count
                new_game_state[b, i, 0] += score_change  # score
                new_game_state[b, i, 1] += 1  # step count
                new_game_state[b, i, 2] += activated_fruits_in_step  # total activated fruits count
        
        return new_inputs_state, actions, new_game_state



class Trainer:
    
    def __init__(self, config: TrainerConfig, device: str):
        self.config = config
        self.brain = GameBrain(config)
        self.engin = GameEngine(config, self.brain)
        self.device = device
    
    def _create_init(self) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns:
            - inits: shape(num_inits, input_size)
        '''
        num_inits = GameConfig.get_initial_num(self.config.game_config)
        
        # Create sprite positions as 1-indexed (1, 2, 3, ..., num_inits)
        sprites = torch.arange(1, num_inits + 1, dtype=torch.float32).unsqueeze(1)
        
        fruits = torch.zeros((num_inits, 3, 3), device=self.device, dtype=torch.float32)
        for idx in range(num_inits):
            # Set first fruit properties for each initialization
            fruits[idx, 0, 0] = random.randint(0, self.config.game_config.screen_width - 1)  # x position
            fruits[idx, 0, 1] = 0  # y position (always starts at 0)
            fruits[idx, 0, 2] = 1.0  # activation status (active)
            # Other fruits remain inactive (already zeros)
        
        # Flatten fruits tensor from (num_inits, 3, 3) to (num_inits, 9)
        fruits_flat = fruits.view(num_inits, -1)
        
        # Concatenate sprite and fruits into a single tensor with shape (num_inits, input_size)
        # input_size = 10, composed of: sprite position (1) + fruits data (3 fruits × 3 dimensions = 9)
        result = torch.cat([sprites, fruits_flat], dim=1)

        # game_state stores the state of each game, each game has 3 dimensions to save its state:
        # index 0 is the score, index 1 indicates number of step when game invoke update, step increse 1.
        # index 2 is the total count of fruits that have been activated during the game
        game_state = torch.zeros((num_inits, 3), dtype=torch.float32, device=self.device)
        
        # Initialize the activated fruits count with 1 for each game (since we start with one active fruit)
        game_state[:, 2] = 1.0  # Each game starts with 1 activated fruit
        
        return result, game_state
    
    def _reward(self, game_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        pass

    def _create_trajector(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
            Creates a bunch of episodes for training. Returns the action, input and reward histories

        Returns:
            - input_history: shape (max_steps, batch_size, num_inits, input_size)
            - action_history: shape (max_steps, batch_size, num_inits)
            - reward_history: shape (max_steps, batch_size, num_inits)
        '''

        num_steps = self.config.max_steps
        batch_size = self.config.batch_size
        num_inits = GameConfig.get_initial_num(self.config.game_config)
        input_size = self.config.game_config.get_inputsize()
        device = self.device

        action_history = torch.zeros((num_steps, batch_size, num_inits), dtype=torch.long, device=device)  # record which word is being used
        score_history = torch.zeros((num_steps, batch_size, num_inits), device=device)  # record the inputs to the neural networks
        input_history = torch.zeros((num_steps, batch_size, num_inits, input_size), device=device)  # record the inputs to the neural networks
        reward_history = torch.zeros((num_steps, batch_size, num_inits), device=device)  # record the rewards

        inits, game_state = self._create_init()
        inputs_state = inits.unsqueeze(0).repeat(batch_size, 1, 1).to(device)  # shape (batch_size, num_inits, input_size)
        game_state  = game_state.unsqueeze(0).repeat(batch_size, 1, 1).to(device)

        for step in range(num_steps):
            # Create input state for the current step
            input_history[step] = inputs_state
            
            # Update the game engine with the current state
            inputs_state, action, game_state = self.engin.update(inputs_state, game_state)
            action_history[step] = action
            
            # Compute rewards and scores from game_state
            reward, score = self._reward(game_state)
            reward_history[step] = reward
            score_history[step] = score
        
        return input_history, action_history, score_history, reward_history