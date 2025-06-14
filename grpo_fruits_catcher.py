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
    min_interval_step_fruits: int = 4
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

    def update(self, inputs_state: torch.Tensor, game_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimized vectorized update of game state for all games in the batch.
        
        Args:
            inputs_state: shape (batch_size, num_inits, input_size) - current game states
            game_state: shape (batch_size, num_inits, 3) - [score, step_count, fruits_reached_bottom_count] for each game
            
        Returns:
            new_inputs_state: updated game states
            actions: chosen actions for each game
            new_game_state: updated game states with scores, step counts, and fruits reached bottom count
        """
        batch_size, num_inits, input_size = inputs_state.shape
        game_config = self.config.game_config
        device = inputs_state.device
        
        # Get actions from brain for all games
        inputs_flat = inputs_state.view(-1, input_size)
        actions, _ = self.brain.sample_action(inputs_flat)
        actions = actions.view(batch_size, num_inits)
        
        # Create new states
        new_inputs_state = inputs_state.clone()
        new_game_state = game_state.clone()
        
        # === VECTORIZED SPRITE MOVEMENT ===
        # Extract sprite positions: shape (batch_size, num_inits)
        sprite_positions = new_inputs_state[:, :, 0].clone()
        
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
        
        # Vectorized interval check: compute minimum distance to y=0 for all games
        min_distances = torch.full((batch_size, num_inits), float('inf'), device=device)
        
        # Calculate minimum distances in batched way
        for fruit_idx in range(max_fruits):
            # Get y positions for this fruit slot across all games
            y_positions = fruit_y[:, :, fruit_idx]  # (batch_size, num_inits)
            is_active = fruit_active[:, :, fruit_idx] == 1.0  # (batch_size, num_inits)
            
            # Calculate distances to y=0 only for active fruits
            distances = torch.abs(y_positions - 0.0)
            
            # Update minimum distances where fruits are active
            min_distances = torch.where(is_active, 
                                      torch.minimum(min_distances, distances), 
                                      min_distances)
        
        # Determine which games can spawn (vectorized)
        can_spawn_mask = (min_distances >= game_config.min_interval_step_fruits) | (min_distances == float('inf'))
        
        # Calculate total fruits needed per game (vectorized)
        random_spawn_needed = should_spawn.int()
        min_fruit_needed = torch.clamp(game_config.min_fruits_on_screen - active_fruit_counts.int(), min=0)
        total_fruits_needed = random_spawn_needed + min_fruit_needed
        
        # Apply interval constraint
        final_fruits_needed = total_fruits_needed * can_spawn_mask.int()
        
        # Spawn fruits efficiently (only loop where needed)
        spawn_indices = (final_fruits_needed > 0).nonzero(as_tuple=False)
        for idx in spawn_indices:
            b, i = idx[0].item(), idx[1].item()
            spawn_count = final_fruits_needed[b, i].item()
            
            # Find inactive slots
            inactive_slots = (fruit_active[b, i] == 0.0).nonzero(as_tuple=True)[0]
            actual_spawn = min(spawn_count, len(inactive_slots))
            
            # Batch spawn fruits
            for j in range(actual_spawn):
                slot = inactive_slots[j]
                fruit_x[b, i, slot] = torch.randint(0, game_config.screen_width, (1,), 
                                                  device=device, dtype=torch.float32)
                fruit_y[b, i, slot] = 0.0
                fruit_active[b, i, slot] = 1.0
        
        # Update fruit data back to the main tensor (create new tensor to avoid memory conflicts)
        fruit_data_flat = torch.stack([fruit_x, fruit_y, fruit_active], dim=3).view(batch_size, num_inits, -1)
        new_inputs_state[:, :, 1:] = fruit_data_flat
        
        # === UPDATE GAME STATE ===
        new_game_state[:, :, 0] += score_changes  # score
        new_game_state[:, :, 1] += 1  # step count
        new_game_state[:, :, 2] += fruits_reached_bottom  # total fruits that reached bottom
        
        return new_inputs_state, actions, new_game_state



class Trainer:
    
    def __init__(self, config: TrainerConfig, device: str):
        self.config = config
        self.device = device
        self.brain = torch.compile(GameBrain(config, device).to(device)) if config.compile else GameBrain(config, device).to(device)
        self.engin = GameEngine(config, self.brain)
        # Improved optimizer configuration
        self.optimizer = torch.optim.AdamW(
            self.brain.parameters(), 
            lr=self.config.lr_rate, 
            betas=(0.9, 0.999), 
            eps=1e-8,
            weight_decay=1e-5  # Small weight decay for regularization
        )
        # Improved learning rate scheduler for more stable training
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=500,  # Reduce learning rate every 500 epochs (more conservative)
            gamma=0.8  # Multiply learning rate by 0.8 (less aggressive reduction)
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
        Calculate reward with improved stability and reduced noise.
        
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
            
            # More balanced reward: +2 for catching fruit, -1 for missing
            catch_reward = torch.clamp(score_delta, min=0.0) * 2.0  # Reduced positive reward
            miss_penalty = torch.clamp(score_delta, max=0.0) * 1.0  # Reduced penalty
            immediate_reward = catch_reward + miss_penalty
            
            # Smaller step penalty to encourage efficiency
            step_penalty = -0.005  # Reduced step penalty
            
            # Smaller baseline reward for stability
            baseline_reward = 0.1  # Reduced baseline
            
            # Progressive score bonus with diminishing returns
            score_bonus = torch.tanh(score * 0.02) * 0.2  # Tanh for stability, reduced scale
            
            reward = immediate_reward + step_penalty + baseline_reward + score_bonus
        else:
            # Initial reward based on current score with smaller scaling
            reward = torch.tanh(score * 0.05) * 0.5  # Tanh for stability
        
        # Significantly reduced noise for better stability
        noise = torch.randn_like(score, device=device) * 0.01  # Reduced noise
        reward = reward + noise
        
        # Clamp to reasonable bounds
        reward = torch.clamp(reward, -5.0, 15.0)
        
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
            
            # Update previous state for next iteration
            prev_game_state = game_state.clone()
        
        return input_history, action_history, score_history, reward_history
    
    def _policy_loss(self, inputs: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor):
        logits = self.brain(inputs)  # Shape: (batch_size, output_size)
        log_probs = F.log_softmax(logits, dim=-1)  # Convert logits to log probabilities
        probs = F.softmax(logits, dim=-1)  # Get probabilities for entropy calculation

        # Extract the log probabilities of the actions taken in the batch
        batch_ix = torch.arange(actions.shape[0], device=self.device)
        log_action_probs = log_probs[batch_ix, actions]  # Correct batch-wise indexing

        # Calculate policy loss with reward clipping for stability
        clipped_rewards = torch.clamp(reward, min=-5.0, max=5.0)  # Clip extreme rewards
        policy_loss = -torch.mean(log_action_probs * clipped_rewards)
        
        # Increased entropy bonus to encourage exploration and prevent overfitting
        entropy = -torch.sum(probs * log_probs, dim=-1)  # Calculate entropy
        entropy_bonus = 0.05 * torch.mean(entropy)  # Increased entropy coefficient
        
        # Add L2 regularization to prevent overfitting
        l2_reg = 0.0001 * sum(p.pow(2.0).sum() for p in self.brain.parameters())
        
        return policy_loss - entropy_bonus + l2_reg  # Include regularization
    
    def _train_epoch(self) -> Tuple[float, float]:
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
        
        # Work backwards to calculate returns with higher discount factor for stability
        discount_factor = 0.95  # Reduced from 0.99 for more stable learning
        for t in reversed(range(num_steps)):
            cumulative_return = reward_history[t] + discount_factor * cumulative_return
            returns[t] = cumulative_return

        # Improved group normalization with better stability
        epsilon = 1e-6  # Smaller epsilon for better numerical stability
        
        # Calculate statistics over all dimensions for more stable normalization
        returns_flat = returns.reshape(-1)
        mean_return = torch.mean(returns_flat)
        std_return = torch.std(returns_flat)
        
        # Apply normalization with minimum std threshold to prevent division by near-zero
        min_std = 0.1
        effective_std = torch.max(std_return, torch.tensor(min_std, device=returns.device))
        
        group_normed_returns = (returns - mean_return) / (effective_std + epsilon)
        
        # Apply additional clipping to prevent extreme values
        group_normed_returns = torch.clamp(group_normed_returns, min=-3.0, max=3.0)
        
        f_returns = group_normed_returns.reshape((num_steps, batch_size * num_inits))
        
        total_loss = 0.0
        for t in range(num_steps):
            # Use the return at time t (not just final reward)
            loss = self._policy_loss(f_input_history[t], f_action_history[t], f_returns[t])
            total_loss += loss

        # Compute gradients and update
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping with more conservative threshold
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=0.5)  # Reduced from 1.0
        
        self.optimizer.step()
        
        # Return average reward and score from the last timestep
        avg_reward = torch.mean(reward_history[-1])
        avg_score = torch.mean(score_history[-1])
        
        return avg_reward.item(), avg_score.item()
    
    def train(self) -> torch.Tensor:
        final_reward = np.zeros(self.config.total_epochs) 
        score = 0.0
        best_score = float('-inf')
        
        # Early stopping parameters
        patience = self.config.patience  # Use patience from config instead of hardcoded value
        no_improvement_count = 0
        best_model_state = None
        
        for epoch in tqdm(range(self.config.total_epochs)):
            final_reward[epoch], score = self._train_epoch()
            
            # Step the learning rate scheduler
            self.scheduler.step()
            
            # Track best performance and implement early stopping
            if score > best_score:
                best_score = score
                no_improvement_count = 0  # Reset counter
                # Save best model state
                best_model_state = {k: v.clone() for k, v in self.brain.state_dict().items()}
                print(f"epoch={epoch}, Reward:{final_reward[epoch]:.6f}, Score: {score:.3f} (NEW BEST!) LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            else:
                no_improvement_count += 1
                if epoch % 50 == 0:  # Print every 50 epochs instead of every 10
                    print(f"epoch={epoch}, Reward:{final_reward[epoch]:.6f}, Score: {score:.3f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping check
            if no_improvement_count >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch}! No improvement for {patience} epochs.")
                print(f"📈 Best score achieved: {best_score:.3f}")
                
                # Restore best model weights
                if best_model_state is not None:
                    self.brain.load_state_dict(best_model_state)
                    print("🔄 Restored best model weights.")
                break
        
        print(f"Final - Reward:{final_reward[epoch]:.6f}, Score: {score:.3f}, Best Score: {best_score:.3f}")
        return final_reward
    
    def save(self, name: str):
        """
        Save the trained model with configs to the specified path.
        Args:
            name (str): Base name for the saved model file
        """
        path = f"{name}-{self.config.total_epochs:06d}.pth"
        
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
        
        torch.save({
            'model_state_dict': self.brain.state_dict(),
            'game_config': game_config_dict,
            'trainer_config': trainer_config_dict,
            'model_class': 'GameBrain'
        }, path)
        print(f"Model saved to {path}")
