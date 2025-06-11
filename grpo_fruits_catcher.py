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
    screen_height: int = 11
    sprite_width: int = 3
    sprite_height: int = 1
    max_fruits_on_screen: int = 3
    min_fruits_on_screen: int = 1

    def get_inputsize(self):
        # Sprite position (1) + fruits data (max_fruits * 3 dimensions)
        return 1 + self.max_fruits_on_screen * 3
    
    @staticmethod
    def get_initial_num(config):
        return config.screen_width


@dataclass
class TrainerConfig:
    hidden_size: int = 256
    batch_size: int = 8
    total_epochs: int = 200
    max_steps: int = 200  # maximum number of steps before game ends
    game_config: Optional[GameConfig] = field(default_factory=GameConfig)
    lr_rate: float = 5e-4
    compile: bool = False


class GameBrain(nn.Module):
    
    def __init__(self, config: TrainerConfig, device: str = 'cpu'):
        super(GameBrain, self).__init__()
        self.device = device
        self.input_size = config.game_config.get_inputsize()
        self.fc_in = nn.Linear(self.input_size, config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc_out = nn.Linear(config.hidden_size, 3)  # 3 actions: left, stay, right
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.constant_(self.fc_in.bias, 0)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc_in(x))
        x = F.relu(self.fc1(x))
        x = self.fc_out(x)
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
        hidden = F.relu(self.fc_in(x))
        
        # Hidden -> Output layer (logits)
        logits = self.fc_out(hidden)
        
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
        
        # Calculate sprite catch boundaries
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
        
        # Find inactive fruit slots and spawn new fruits
        for b in range(batch_size):
            for i in range(num_inits):
                if should_spawn[b, i]:
                    # Find first inactive fruit slot
                    inactive_slots = (fruit_active[b, i] == 0.0).nonzero(as_tuple=True)[0]
                    if len(inactive_slots) > 0:
                        slot = inactive_slots[0]
                        fruit_x[b, i, slot] = torch.randint(0, game_config.screen_width, (1,), 
                                                          device=device, dtype=torch.float32)
                        fruit_y[b, i, slot] = 0.0
                        fruit_active[b, i, slot] = 1.0
        
        # Ensure minimum fruits are active
        needs_more_fruits = active_fruit_counts < game_config.min_fruits_on_screen
        for b in range(batch_size):
            for i in range(num_inits):
                if needs_more_fruits[b, i]:
                    needed = int(game_config.min_fruits_on_screen - active_fruit_counts[b, i].item())
                    inactive_slots = (fruit_active[b, i] == 0.0).nonzero(as_tuple=True)[0]
                    for j in range(min(needed, len(inactive_slots))):
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
        # Fix: Create proper optimizer instead of just parameters
        #self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=self.config.lr_rate)
        self.optimizer = torch.optim.AdamW(self.brain.parameters(), 
                                           lr=self.config.lr_rate, 
                                           betas=(0.9, 0.95), 
                                           eps=1e-8)

    
    def _create_init(self) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns:
            - inits: shape(num_inits, input_size)
        '''
        num_inits = GameConfig.get_initial_num(self.config.game_config)
        
        # Create sprite positions as 1-indexed (1, 2, 3, ..., num_inits)
        sprites = torch.arange(1, num_inits + 1, dtype=torch.float32, device=self.device).unsqueeze(1)
        
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
        Calculate reward based on game state using exponential catch rate bonus.
        
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
        
        # Estimate caught fruits from score and total fruits
        # Assuming: score = caught_fruits - missed_fruits = caught_fruits - (total_fruits - caught_fruits) = 2*caught_fruits - total_fruits
        # So: caught_fruits = (score + total_fruits) / 2
        total_fruits = fruits_reached_bottom
        caught_fruits = torch.clamp((score + total_fruits) / 2.0, min=0.0)
        caught_fruits = torch.min(caught_fruits, total_fruits)  # Ensure caught_fruits <= total_fruits
        
        # Calculate catch rate
        catch_rate = caught_fruits / (total_fruits + 1e-6)
        
        # Exponential bonus for catch rate - heavily rewards high performance  
        catch_bonus = torch.exp(catch_rate * 2.0) - 1.0  # Range: 0 to ~6.4
        
        # Score-based component (normalized)
        score_component = score * 0.1
        
        # Survival bonus - reward staying alive longer
        survival_bonus = 0.1
        
        # Penalty for poor performance (negative scores)
        penalty = torch.clamp(score * -0.05, max=0.0)
        
        # Add small random noise to ensure group variance for GRPO
        noise = torch.randn_like(score, device=device) * 0.1
        
        # Combine components
        reward = catch_bonus + score_component + survival_bonus + penalty + noise
        
        # Clamp to reasonable bounds
        reward = torch.clamp(reward, -5.0, 10.0)
        
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

        # Extract the log probabilities of the actions taken in the batch
        batch_ix = torch.arange(actions.shape[0], device=self.device)
        log_action_probs = log_probs[batch_ix, actions]  # Correct batch-wise indexing to get the log prob of element i in index i

        return -torch.mean(log_action_probs * reward)  # Scalar loss averaged over the batch
    
    def _train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch using GRPO (Group Reward Policy Optimization).
        
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
        last_reward = torch.mean(reward_history[-1])

        epsilon = 1e-7  # for safe division by the STD
        group_normed_reward_history = (reward_history - torch.mean(reward_history, dim=-1, keepdim=True)) / (torch.std(reward_history, dim=-1, keepdim=True) + epsilon)  # THE GRPO LINE!!!
        f_reward_history = group_normed_reward_history.reshape((num_steps, batch_size * num_inits))
        total_loss = 0.0
        for t in range(num_steps):
            # Compute the gradient at time step t
            f_return = f_reward_history[-1]  # the reward is the reward at the FINAL time only

            loss = self._policy_loss(f_input_history[t], f_action_history[t], f_return)
            total_loss += loss

        # Compute gradients and update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return last_reward.item(), torch.mean(score_history[-1]).item()
    
    def train(self) -> torch.Tensor:
        final_reward = np.zeros(self.config.total_epochs) 
        for epoch in tqdm(range(self.config.total_epochs)):
            final_reward[epoch], score = self._train_epoch()
            print(f"{epoch=}, Reward:{final_reward[epoch]:.6f}, Score: {score:.3f}")
        
        return final_reward
    
    def save(self, name: str):
        """
        Save the trained model to the specified path.
        Args:
            path (str): Path to save the model
        """
        path = f"{name}-{self.config.total_epochs:06d}.pth"
        torch.save(self.brain.state_dict(), path)
