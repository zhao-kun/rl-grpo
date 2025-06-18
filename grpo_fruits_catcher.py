from dataclasses import dataclass, field
from typing import Optional, Tuple
from torch import nn

import torch
import torch.nn.functional as F 
import random
import numpy as np
from tqdm import tqdm

@dataclass
class GameConfig:
    """Configuration for the fruit catching game"""
    screen_width: int = 20
    screen_height: int = 15
    sprite_width: int = 3
    sprite_height: int = 1
    max_fruits_on_screen: int = 3
    min_fruits_on_screen: int = 1
    min_interval_step_fruits: int = 5
    view_height_multiplier: float = 50.0  #  Use to scale the height for view cordinate to get better visual effect
    view_width_multiplier: float = 50.0  #  Use to scale the width for view cordinate to get better visual effect
    refresh_timer: int = 150  # When refresh_timer reach, the game game engine will update once, in millisecond
    fail_ended_game_score: int = -30 # Game ends when score reaches this value
    win_ended_game_score: int = 30

    def get_inputsize(self):
        # Sprite position (1) + fruits data (max_fruits * 3 dimensions)
        return 1 + self.max_fruits_on_screen * 3
    
    @staticmethod
    def get_initial_num(config):
        return config.screen_width - 3


@dataclass
class TrainerConfig:
    hidden_size: int = 512
    batch_size: int = 8
    total_epochs: int = 200
    max_steps: int = 200  # maximum number of steps before game ends
    game_config: Optional[GameConfig] = field(default_factory=GameConfig)
    lr_rate: float = 5e-4
    compile: bool = False
    patience: int = 300  # Number of epochs to wait for improvement before early stopping
    save_checkpoint_per_num_epoch: int = 50  # Save the checkpoint per number of epochs, -1 is no save (only save in the end)
    save_best_model: bool = True  # Save the best model during the training, based on best score achieved
    model_name: str = "grpo_fruits_catcher"  # Name of the model to save


class GameBrain(nn.Module):
    
    def __init__(self, config: TrainerConfig, device: str = 'cpu'):
        super(GameBrain, self).__init__()
        self.device = device
        self.input_size = config.game_config.get_inputsize()
        self.fc_in = nn.Linear(self.input_size, config.hidden_size, bias=False)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization
        self.fc_out = nn.Linear(config.hidden_size, 3)  # 3 actions: left, stay, right
        self._init_weights()
        self.gelu = nn.GELU(approximate='tanh')
        self.layer_norm = nn.LayerNorm(config.hidden_size)  # Layer normalization for stability

    def _init_weights(self):
        """Initialize network weights with better initialization"""
        # Use He initialization for ReLU networks
        nn.init.normal_(self.fc_in.weight, mean=0.0, std=0.02)
        #nn.init.kaiming_normal_(self.fc_in.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.constant_(self.fc_in.bias, 0)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        # Small initialization for output layer to prevent extreme initial policy
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Action logits of shape (batch_size, output_size)
        """
        # Input -> Hidden layer with ReLU activation
        x = self.fc_in(x)
        x = x + self.layer_norm(self.gelu(x))
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        logits = self.fc_out(x)
        
        return F.log_softmax(logits, dim=-1)
    
    def sample_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Args:
            x: Input state tensor
            
        Returns:
            (action, log_probability)
        """
        log_probs = self.forward(x)
        #probs = F.softmax(logits, dim=-1)
        
        # Sample action from probability distribution
        action_dist = torch.distributions.Categorical(logits=log_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = 'cpu') -> Tuple['GameBrain', 'GameConfig', 'TrainerConfig']:
        """
        Load a pretrained GameBrain model with its configurations.
        
        Args:
            path: Path to the saved model file
            device: Device to load the model on
            
        Returns:
            Tuple of (GameBrain instance, GameConfig, TrainerConfig)
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Check if this is a new format checkpoint with configs or old format with just weights
        if 'game_config' in checkpoint and 'trainer_config' in checkpoint:
            # New format with saved configs
            # Reconstruct GameConfig from saved dictionary
            game_config_dict = checkpoint['game_config']
            game_config = GameConfig(
                screen_width=game_config_dict['screen_width'],
                screen_height=game_config_dict['screen_height'],
                sprite_width=game_config_dict['sprite_width'],
                sprite_height=game_config_dict['sprite_height'],
                max_fruits_on_screen=game_config_dict['max_fruits_on_screen'],
                min_fruits_on_screen=game_config_dict['min_fruits_on_screen'],
                min_interval_step_fruits=game_config_dict['min_interval_step_fruits'],
                view_height_multiplier=game_config_dict['view_height_multiplier'],
                view_width_multiplier=game_config_dict['view_width_multiplier'],
                refresh_timer=game_config_dict['refresh_timer'],
                fail_ended_game_score=game_config_dict.get('fail_ended_game_score', -30),
                win_ended_game_score=game_config_dict.get('win_ended_game_score', 30)
            )
            
            # Reconstruct TrainerConfig from saved dictionary
            trainer_config_dict = checkpoint['trainer_config']
            trainer_config = TrainerConfig(
                hidden_size=trainer_config_dict['hidden_size'],
                batch_size=trainer_config_dict['batch_size'],
                total_epochs=trainer_config_dict['total_epochs'],
                max_steps=trainer_config_dict['max_steps'],
                game_config=game_config,
                lr_rate=trainer_config_dict['lr_rate'],
                compile=trainer_config_dict['compile'],
                patience=trainer_config_dict.get('patience', 300)  # Default to 300 for backward compatibility
            )
            
            # Create and load the model
            model = cls(trainer_config, device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"Model loaded from {path} (new format)")
        else:
            # Old format with just model weights - use default configs
            print(f"Loading model from {path} (old format - using default configs)")
            
            # Use default configurations
            game_config = GameConfig()
            trainer_config = TrainerConfig(game_config=game_config)
            
            # Create model with default config
            model = cls(trainer_config, device)
            
            # Load the state dict directly (old format)
            model.load_state_dict(checkpoint)
        
        model = model.eval().to(device=device)  # Set to evaluation mode
        
        print(f"Game config: {game_config.screen_width}x{game_config.screen_height}, max_fruits: {game_config.max_fruits_on_screen}")
        print(f"Model config: hidden_size={trainer_config.hidden_size}")
        print(f"Win/Lose thresholds: {game_config.win_ended_game_score}/{game_config.fail_ended_game_score}")
        
        return model, game_config, trainer_config


class GameEngine:
    
    def __init__(self, config: TrainerConfig, brain: GameBrain ):
        self.config = config
        self.brain = brain
    
    # OPTIMIZATION: Prepare for torch.compile optimization of update method
    # Note: torch.compile works best when applied selectively to computational bottlenecks

    def update(self, inputs_state: torch.Tensor, game_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimized vectorized update of game state for all games in the batch.
        Uses torch.compile for better performance on the computational hot path.
        
        Args:
            inputs_state: shape (batch_size, num_inits, input_size) - current game states
            game_state: shape (batch_size, num_inits, 3) - [score, step_count, fruits_reached_bottom_count] for each game
            
        Returns:
            new_inputs_state: updated game states
            actions: chosen actions for each game
            new_game_state: updated game states with scores, step counts, and fruits reached bottom count
        """
        # OPTIMIZATION: torch.compile is applied to GameBrain, which provides the main speedup
        batch_size, num_inits, input_size = inputs_state.shape
        game_config = self.config.game_config
        device = inputs_state.device
        
        # Get actions from brain for all games
        inputs_flat = inputs_state.view(-1, input_size)
        actions, _ = self.brain.sample_action(inputs_flat)
        actions = actions.view(batch_size, num_inits)
        
        # OPTIMIZATION: Avoid unnecessary memory allocations - use detach for no-grad operations
        new_inputs_state = inputs_state.detach().clone()
        new_game_state = game_state.detach().clone()
        
        # === VECTORIZED SPRITE MOVEMENT ===
        # Extract sprite positions: shape (batch_size, num_inits)
        sprite_positions = new_inputs_state[:, :, 0]
        
        # Vectorized sprite movement based on actions
        # action 0=left, 1=stay, 2=right
        left_mask = (actions == 0)
        right_mask = (actions == 2)
        
        sprite_positions[left_mask] = torch.clamp(sprite_positions[left_mask] - 1, min=0)
        sprite_positions[right_mask] = torch.clamp(sprite_positions[right_mask] + 1, 
                                                  max=game_config.screen_width - 1)
        
        new_inputs_state[:, :, 0] = sprite_positions
        
        # === VECTORIZED FRUIT PROCESSING ===
        max_fruits = game_config.max_fruits_on_screen
        
        # Reshape fruit data for easier processing: (batch_size, num_inits, max_fruits, 3)
        fruit_data = new_inputs_state[:, :, 1:].view(batch_size, num_inits, max_fruits, 3)
        
        # Extract fruit properties
        fruit_x = fruit_data[:, :, :, 0]  # x positions
        fruit_y = fruit_data[:, :, :, 1]  # y positions  
        fruit_active = fruit_data[:, :, :, 2]  # activation status
        
        # Move all active fruits down by 1
        active_mask = (fruit_active == 1.0)
        fruit_y[active_mask] += 1
        
        # Check collisions and scoring
        sprite_y = game_config.screen_height - 1
        bottom_reached_mask = (fruit_y == sprite_y) & active_mask
        off_screen_mask = (fruit_y >= game_config.screen_height) & active_mask
        
        # Calculate sGroup Relative Policy Optimizationprite catch boundaries
        sprite_left = sprite_positions.unsqueeze(2) - game_config.sprite_width // 2  # (batch_size, num_inits, 1)
        sprite_right = sprite_positions.unsqueeze(2) + game_config.sprite_width // 2
        
        # Check which fruits are caught
        caught_mask = (bottom_reached_mask & 
                      (fruit_x >= sprite_left) & 
                      (fruit_x <= sprite_right))
        
        # Check which fruits are missed (reached bottom but not caught)
        missed_mask = bottom_reached_mask & ~caught_mask
        
        # Deactivate fruits that reached bottom or went off screen
        deactivate_mask = bottom_reached_mask | off_screen_mask
        fruit_active[deactivate_mask] = 0.0
        
        # Calculate score changes
        score_changes = caught_mask.sum(dim=2).float() - missed_mask.sum(dim=2).float()  # (batch_size, num_inits)
        fruits_reached_bottom = (bottom_reached_mask | off_screen_mask).sum(dim=2).float()
        fruits_deactivated = deactivate_mask.sum(dim=2).float()
        
        # === VECTORIZED FRUIT SPAWNING ===
        # Count active fruits after processing
        active_fruit_counts = fruit_active.sum(dim=2)  # (batch_size, num_inits)
        
        # Conditions for spawning new fruits
        can_spawn = ((fruits_deactivated == 0) & 
                    (active_fruit_counts < game_config.max_fruits_on_screen) &
                    (active_fruit_counts >= game_config.min_fruits_on_screen))
        
        # Random decision to spawn (50% chance)
        spawn_random = torch.rand(batch_size, num_inits, device=device) > 0.5
        should_spawn = can_spawn & spawn_random
        
        # === OPTIMIZED VECTORIZED FRUIT SPAWNING (O(n) complexity) ===
        needs_more_fruits = active_fruit_counts < game_config.min_fruits_on_screen
        
        # OPTIMIZATION: Fully vectorized minimum distance calculation
        # Calculate all distances at once without loops
        # fruit_y shape: (batch_size, num_inits, max_fruits)
        # fruit_active shape: (batch_size, num_inits, max_fruits)
        
        # Calculate distances for all fruits at once
        all_distances = torch.abs(fruit_y - 0.0)  # Distance to y=0 for all fruits
        
        # Mask inactive fruits with large values so they don't affect minimum
        masked_distances = torch.where(fruit_active == 1.0, 
                                     all_distances, 
                                     torch.full_like(all_distances, float('inf')))
        
        # Find minimum distance across all fruits for each game
        min_distances, _ = torch.min(masked_distances, dim=2)  # (batch_size, num_inits)
        
        # Determine which games can spawn (vectorized)
        can_spawn_mask = (min_distances >= game_config.min_interval_step_fruits) | (min_distances == float('inf'))
        
        # Calculate total fruits needed per game (vectorized)
        random_spawn_needed = should_spawn.int()
        min_fruit_needed = torch.clamp(game_config.min_fruits_on_screen - active_fruit_counts.int(), min=0)
        total_fruits_needed = random_spawn_needed + min_fruit_needed
        
        # Apply interval constraint
        final_fruits_needed = total_fruits_needed * can_spawn_mask.int()
        
        # OPTIMIZATION: Vectorized fruit spawning to eliminate GPU-CPU synchronization
        spawn_mask = final_fruits_needed > 0
        if spawn_mask.any():
            # Pre-compute all spawn positions to avoid individual tensor creation
            total_spawns_needed = final_fruits_needed.sum().item()
            if total_spawns_needed > 0:
                # Generate all random positions at once
                spawn_x_positions = torch.randint(0, game_config.screen_width, 
                                                (total_spawns_needed,), 
                                                device=device, dtype=torch.float32)
                
                spawn_counter = 0
                # Process only games that need spawning
                spawn_indices = spawn_mask.nonzero(as_tuple=False)
                for idx in spawn_indices:
                    b, i = idx[0].item(), idx[1].item()
                    spawn_count = final_fruits_needed[b, i].item()
                    
                    # Find inactive slots (vectorized)
                    inactive_mask = fruit_active[b, i] == 0.0
                    inactive_slots = inactive_mask.nonzero(as_tuple=True)[0]
                    actual_spawn = min(spawn_count, len(inactive_slots))
                    
                    # Assign pre-computed positions
                    for j in range(actual_spawn):
                        slot = inactive_slots[j]
                        fruit_x[b, i, slot] = spawn_x_positions[spawn_counter]
                        fruit_y[b, i, slot] = 0.0
                        fruit_active[b, i, slot] = 1.0
                        spawn_counter += 1
        
        # OPTIMIZATION: More efficient tensor reconstruction - avoid stack operation
        # Directly update the slice instead of creating intermediate tensors
        fruit_data_reshaped = torch.cat([fruit_x.unsqueeze(3), fruit_y.unsqueeze(3), fruit_active.unsqueeze(3)], dim=3)
        new_inputs_state[:, :, 1:] = fruit_data_reshaped.view(batch_size, num_inits, -1)
        
        # === UPDATE GAME STATE ===
        new_game_state[:, :, 0] += score_changes  # score
        new_game_state[:, :, 1] += 1  # step count
        new_game_state[:, :, 2] += fruits_reached_bottom  # total fruits that reached bottom
        
        return new_inputs_state, actions, new_game_state



class Trainer:
    
    def __init__(self, config: TrainerConfig, device: str):
        self.config = config
        self.device = device
        # OPTIMIZATION: Apply torch.compile to GameBrain for better training performance
        # With robust fallback for older GPUs that don't support torch.compile
        self.brain = GameBrain(config, device).to(device)
        
        if config.compile:
            try:
                # Test if torch.compile works by checking GPU capability first
                if torch.cuda.is_available():
                    capability = torch.cuda.get_device_capability(device)
                    if capability[0] < 7:  # Triton requires compute capability >= 7.0
                        print(f"⚠️  GPU compute capability {capability[0]}.{capability[1]} < 7.0, skipping torch.compile")
                        self._compile_enabled = False
                    else:
                        self.brain = torch.compile(self.brain)
                        print("🚀 Using torch.compile optimizations for faster training")
                        self._compile_enabled = True
                else:
                    # CPU compilation
                    self.brain = torch.compile(self.brain)
                    print("🚀 Using torch.compile optimizations for faster training (CPU)")
                    self._compile_enabled = True
            except Exception as e:
                print(f"⚠️  torch.compile failed: {str(e)[:50]}..., using eager mode")
                self._compile_enabled = False
        else:
            self._compile_enabled = False
        
        self.engin = GameEngine(config, self.brain)
        # OPTIMIZATION: Improved optimizer configuration with fused operations when available
        self.optimizer = torch.optim.AdamW(
            self.brain.parameters(), 
            lr=self.config.lr_rate, 
            betas=(0.9, 0.999), 
            eps=1e-8,
            weight_decay=1e-5,  # Small weight decay for regularization
            fused=torch.cuda.is_available()  # Use fused AdamW for better GPU performance
        )
        # Improved learning rate scheduler for more stable training
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=1000,  # Reduce learning rate every 1000 epochs (more conservative)
            gamma=0.95  # Multiply learning rate by 0.95 (less aggressive reduction)
        )

    
    def _create_init(self) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns:
            - inits: shape(num_inits, input_size)
        '''
        num_inits = GameConfig.get_initial_num(self.config.game_config)
        
        # Create sprite positions as 1-indexed (1, 2, 3, ..., num_inits)
        sprites = torch.arange(1, num_inits+1, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        max_fruits = self.config.game_config.max_fruits_on_screen
        fruits = torch.zeros((num_inits, max_fruits, 3), device=self.device, dtype=torch.float32)
        for idx in range(num_inits):
            # Set first fruit properties for each initialization
            fruits[idx, 0, 0] = random.randint(0, self.config.game_config.screen_width - 1)  # x position
            fruits[idx, 0, 1] = 0  # y position (always starts at 0)
            fruits[idx, 0, 2] = 1.0  # activation status (active)
            # Other fruits remain inactive (already zeros)
        
        # Flatten fruits tensor from (num_inits, max_fruits, 3) to (num_inits, max_fruits * 3)
        fruits_flat = fruits.view(num_inits, -1)
        
        # Concatenate sprite and fruits into a single tensor with shape (num_inits, input_size)
        # input_size = 1 + max_fruits * 3, composed of: sprite position (1) + fruits data (max_fruits × 3 dimensions)
        result = torch.cat([sprites, fruits_flat], dim=1)

        # game_state stores the state of each game, each game has 3 dimensions to save its state:
        # index 0 is the score, index 1 indicates number of step when game invoke update, step increse 1.
        # index 2 is the total count of fruits that have reached the bottom during the game
        game_state = torch.zeros((num_inits, 3), dtype=torch.float32, device=self.device)
        
        # Initialize the fruits reached bottom count with 0 for each game (no fruits have reached bottom yet)
        # game_state[:, 2] = 0.0  # Already 0 from torch.zeros, but explicit for clarity
        
        return result, game_state
    
    def _reward(self, game_state: torch.Tensor, prev_game_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate reward with improved stability and better scaling.
        
        Args:
            game_state: shape (batch_size, num_inits, 3) containing [score, step_count, fruits_reached_bottom]
            prev_game_state: previous game state for calculating incremental rewards
            
        Returns:
            reward: shape (batch_size, num_inits) - reward for current state
            score: shape (batch_size, num_inits) - score component for tracking
        """
        batch_size, num_inits, _ = game_state.shape
        device = game_state.device
        
        # Extract components from current game state
        score = game_state[:, :, 0]  # cumulative score (caught fruits - missed fruits)
        step_count = game_state[:, :, 1]  # number of steps taken
        fruits_reached_bottom = game_state[:, :, 2]  # total fruits that reached bottom
        
        # Calculate incremental score change if previous state is available
        if prev_game_state is not None:
            prev_score = prev_game_state[:, :, 0]
            score_delta = score - prev_score
            
            # Immediate reward for actions: +1 for catching, -0.5 for missing
            catch_reward = torch.clamp(score_delta, min=0.0) * 1.0
            miss_penalty = torch.clamp(score_delta, max=0.0) * 0.5
            immediate_reward = catch_reward + miss_penalty
            
            # Very small step penalty to encourage efficiency without dominating
            step_penalty = -0.01
            
            # Small baseline to keep rewards positive on average
            baseline_reward = 0.05
            
            # Scale the total reward to reasonable range
            reward = immediate_reward + step_penalty + baseline_reward
        else:
            # Calculate reward based on current performance metrics
            # Avoid division by zero
            safe_fruits_reached = torch.clamp(fruits_reached_bottom, min=1.0)
            catch_rate = torch.clamp(score, min=0.0) / safe_fruits_reached
            
            # Exponential bonus for high catch rates
            catch_rate_bonus = torch.exp(catch_rate) - 1.0
            
            # Performance-based reward
            performance_reward = score * 0.1 + catch_rate_bonus * 0.5
            
            # Small baseline to keep rewards positive on average
            baseline_reward = 0.05
            
            # Combine rewards
            reward = performance_reward + baseline_reward
        
        # Remove noise completely for more stable training
        # noise = torch.randn_like(score, device=device) * 0.01
        # reward = reward + noise
        
        # Clamp to prevent extreme values
        reward = torch.clamp(reward, -2.0, 2.0)
        
        return reward, score


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
        prev_game_state = None  # Track previous state for incremental rewards

        for step in range(num_steps):
            # Create input state for the current step
            input_history[step] = inputs_state
            
            # Update the game engine with the current state
            inputs_state, action, game_state = self.engin.update(inputs_state, game_state)
            action_history[step] = action
            
            # Compute rewards and scores from game_state with previous state context
            reward, score = self._reward(game_state, prev_game_state)
            reward_history[step] = reward
            score_history[step] = score
            
            # OPTIMIZATION: Use detach to avoid unnecessary gradient tracking
            prev_game_state = game_state.detach().clone()
        
        return input_history, action_history, score_history, reward_history
    
    def _policy_loss(self, inputs: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor):
        """
        Calculate policy loss with proper log probability handling.
        
        Args:
            inputs: Game state inputs
            actions: Actions taken
            reward: Normalized rewards/returns
            
        Returns:
            Policy loss tensor
        """
        log_probs = self.brain(inputs)  # Shape: (batch_size, output_size) - log probabilities
        
        # Extract the log probabilities of the actions taken in the batch
        batch_ix = torch.arange(actions.shape[0], device=self.device)
        log_action_probs = log_probs[batch_ix, actions]  # These are already log probabilities
        
        # Calculate policy loss with reward clipping for stability
        reward = torch.clamp(reward, min=-5.0, max=5.0)  # More conservative clipping
        
        # REINFORCE loss: -log(π(a|s)) * R
        # Since we already have log probabilities, we use them directly
        policy_loss = -torch.mean(log_action_probs * reward)
        
        # OPTIMIZATION: Direct entropy calculation from log probabilities
        # Instead of exp(log_probs) * log_probs, use direct calculation
        entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)
        entropy_bonus = 0.01 * torch.mean(entropy)  # Entropy coefficient
        
        return policy_loss - entropy_bonus
    
    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Train for one epoch using GRPO with cumulative returns.
        
        Returns:
            Tuple[float, float]: (average_reward, average_score) for the epoch
        """

        num_steps = self.config.max_steps
        batch_size = self.config.batch_size
        num_inits = GameConfig.get_initial_num(self.config.game_config)
        input_size = self.config.game_config.get_inputsize()

        input_history, action_history, score_history, reward_history = self._create_trajector()
        f_input_history = input_history.reshape((num_steps, batch_size * num_inits, input_size))
        f_action_history = action_history.reshape((num_steps, batch_size * num_inits))
        
        # Calculate cumulative returns (sum of future rewards) for each timestep
        returns = torch.zeros_like(reward_history)
        cumulative_return = torch.zeros_like(reward_history[-1])
        
        # Work backwards to calculate returns with appropriate discount factor
        discount_factor = 0.99  # Restored higher discount for better long-term learning
        for t in reversed(range(num_steps)):
            cumulative_return = reward_history[t] + discount_factor * cumulative_return
            returns[t] = cumulative_return

        # Improved return normalization with better stability
        epsilon = 1e-6
        
        # Calculate statistics over all dimensions
        returns_flat = returns.reshape(-1)
        mean_return = torch.mean(returns_flat)
        std_return = torch.std(returns_flat)
        
        # Apply less aggressive normalization to prevent extreme values
        # Keep some of the original scale to prevent loss from becoming too negative
        min_std = 0.5  # Higher minimum std for stability
        effective_std = torch.max(std_return, torch.tensor(min_std, device=returns.device))
        
        # Use gentler normalization that preserves some original scale
        group_normed_returns = (returns - mean_return * 0.5) / (effective_std + epsilon)
        
        # Apply conservative clipping to prevent extreme normalized values
        group_normed_returns = torch.clamp(group_normed_returns, min=-2.0, max=2.0)
        
        f_returns = group_normed_returns.reshape((num_steps, batch_size * num_inits))
        
        # OPTIMIZATION: Fix CPU tensor allocation - create on GPU device
        total_loss = torch.zeros(num_steps, device=self.device)
        for t in range(num_steps):
            # Use the return at time t (not just final reward)
            loss = self._policy_loss(f_input_history[t], f_action_history[t], f_returns[t])
            total_loss[t] = loss

        # Compute gradients and update
        total_loss = total_loss.mean()  # Average loss over all timesteps
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError("Loss is NaN or Inf, indicating instability in training.")

        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Conservative gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=1.0)  # Restored reasonable clipping
        
        self.optimizer.step()
        
        # Return average reward and score from the last timestep
        avg_reward = torch.mean(reward_history[-1])
        avg_score = torch.mean(score_history[-1])
        
        return avg_reward.item(), avg_score.item(), total_loss.item()
    
    def train(self) -> torch.Tensor:
        final_reward = np.zeros(self.config.total_epochs) 
        score = 0.0
        best_score = float('-inf')
        
        # Early stopping parameters
        patience = self.config.patience  # Use patience from config instead of hardcoded value
        no_improvement_count = 0
        best_model_state = None
        best_epoch = 0
        
        for epoch in tqdm(range(self.config.total_epochs)):
            final_reward[epoch], score, loss = self._train_epoch()
            
            # Step the learning rate scheduler
            self.scheduler.step()
            
            # Track best performance and implement early stopping
            if score > best_score:
                best_score = score
                no_improvement_count = 0  # Reset counter
                # Save best model state
                best_model_state = {k: v.clone() for k, v in self.brain.state_dict().items()}
                best_epoch = epoch
                print(f"epoch={epoch}, Reward:{final_reward[epoch]:.6f}, Score: {score:.3f} (NEW BEST!) LR: {self.optimizer.param_groups[0]['lr']:.2e}, Loss: {loss:.4f}")
            else:
                no_improvement_count += 1
                if epoch % 50 == 0:  # Print every 50 epochs instead of every 10
                    print(f"epoch={epoch}, Reward:{final_reward[epoch]:.6f}, Score: {score:.3f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}, Loss: {loss:.4f}")

            if self.config.save_checkpoint_per_num_epoch > 0 and epoch % self.config.save_checkpoint_per_num_epoch == 0 and epoch > 0:
                # Save checkpoint every `save_checkpoint_per_num_epoch` epochs
                self._save_model(f"{self.config.model_name}", self.brain, current_epochs=epoch)
                print(f"Checkpoint saved at epoch {epoch}")

            # Early stopping check
            if no_improvement_count >= patience and patience > 0:
                print(f"\n🛑 Early stopping at epoch {epoch}! No improvement for {patience} epochs.")
                print(f"📈 Best score achieved: {best_score:.3f}")
                break
                
                # Restore best model weights
        if best_model_state is not None and best_epoch < epoch:
            gb = GameBrain(self.config)
            gb.load_state_dict(best_model_state)
            self._save_model(f"{self.config.model_name}-best", gb, current_epochs=best_epoch)
            print("🔄 Saved best model weights.")

        print(f"Final - Reward:{final_reward[epoch]:.6f}, Score: {score:.3f}, Best Score: {best_score:.3f}")
        return final_reward
    
    def save(self, name: str):
        """
        Save the trained model with configs to the specified path.
        Args:
            name (str): Base name for the saved model file
        """
        # Convert configs to dictionaries for JSON serialization
        game_config_dict = {
            'screen_width': self.config.game_config.screen_width,
            'screen_height': self.config.game_config.screen_height,
            'sprite_width': self.config.game_config.sprite_width,
            'sprite_height': self.config.game_config.sprite_height,
            'max_fruits_on_screen': self.config.game_config.max_fruits_on_screen,
            'min_fruits_on_screen': self.config.game_config.min_fruits_on_screen,
            'min_interval_step_fruits': self.config.game_config.min_interval_step_fruits,
            'view_height_multiplier': self.config.game_config.view_height_multiplier,
            'view_width_multiplier': self.config.game_config.view_width_multiplier,
            'refresh_timer': self.config.game_config.refresh_timer,
            'fail_ended_game_score': self.config.game_config.fail_ended_game_score,
            'win_ended_game_score': self.config.game_config.win_ended_game_score
        }
        
        trainer_config_dict = {
            'hidden_size': self.config.hidden_size,
            'batch_size': self.config.batch_size,
            'total_epochs': self.config.total_epochs,
            'max_steps': self.config.max_steps,
            'lr_rate': self.config.lr_rate,
            'compile': self.config.compile,
            'patience': self.config.patience
        }
        
        self.__save(name, game_config_dict, trainer_config_dict, self.brain)

    def _save_model(self, name: str, model: nn.Module, current_epochs: int = None):
        """
        Save the model state and configurations to a file.
        
        Args:
            name (str): Base name for the saved model file
            trainer_config (TrainerConfig): Trainer configuration object
        """
        game_config_dict = {
            'screen_width': self.config.game_config.screen_width,
            'screen_height': self.config.game_config.screen_height,
            'sprite_width': self.config.game_config.sprite_width,
            'sprite_height': self.config.game_config.sprite_height,
            'max_fruits_on_screen': self.config.game_config.max_fruits_on_screen,
            'min_fruits_on_screen': self.config.game_config.min_fruits_on_screen,
            'min_interval_step_fruits': self.config.game_config.min_interval_step_fruits,
            'view_height_multiplier': self.config.game_config.view_height_multiplier,
            'view_width_multiplier': self.config.game_config.view_width_multiplier,
            'refresh_timer': self.config.game_config.refresh_timer,
            'fail_ended_game_score': self.config.game_config.fail_ended_game_score,
            'win_ended_game_score': self.config.game_config.win_ended_game_score
        }
        
        trainer_config_dict = {
            'hidden_size': self.config.hidden_size,
            'batch_size': self.config.batch_size,
            'total_epochs': self.config.total_epochs,
            'max_steps': self.config.max_steps,
            'lr_rate': self.config.lr_rate,
            'compile': self.config.compile,
            'patience': self.config.patience
        }
        
        self.__save(name, game_config_dict, trainer_config_dict, model, current_epochs)

    def __save(self, name, game_config_dict: dict, trainer_config_dict: dict, model: nn.Module, current_epochs: int = None):
        """
        Save the model state and configurations to a file.
        
        Args:
            name (str): Base name for the saved model file
            game_config_dict (dict): Game configuration dictionary
            trainer_config_dict (dict): Trainer configuration dictionary
        """
        path = f"{name}-{self.config.total_epochs:06d}-{current_epochs:06d}.pth" if current_epochs else f"{name}-{self.config.total_epochs:06d}.pth"

        torch.save({
            'model_state_dict': model.state_dict(),
            'game_config': game_config_dict,
            'trainer_config': trainer_config_dict,
            'model_class': 'GameBrain'
        }, path)
        print(f"Model saved to {path}")
    
