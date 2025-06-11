from dataclasses import dataclass, field
from typing import Optional, Tuple
from torch import nn
import np
import tqdm

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
            game_state: shape (batch_size, num_inits, 3) - [score, step_count, fruits_reached_bottom_count] for each game
            
        Returns:
            new_inputs_state: updated game states
            actions: chosen actions for each game
            new_game_state: updated game states with scores, step counts, and fruits reached bottom count
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
                fruits_reached_bottom_count = 0
                
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
                            # Fruit reached the bottom level
                            fruits_reached_bottom_count += 1
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
                            # Fruit went off screen (also counts as reaching bottom)
                            fruits_reached_bottom_count += 1
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
                
                # Update score, step count, and fruits reached bottom count
                new_game_state[b, i, 0] += score_change  # score
                new_game_state[b, i, 1] += 1  # step count
                new_game_state[b, i, 2] += fruits_reached_bottom_count  # total fruits that reached bottom
        
        return new_inputs_state, actions, new_game_state



class Trainer:
    
    def __init__(self, config: TrainerConfig, device: str):
        self.config = config
        self.brain = GameBrain(config)
        self.engin = GameEngine(config, self.brain)
        self.device = device
        self.optimizer = self.brain.parameters()
    
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
        # index 2 is the total count of fruits that have reached the bottom during the game
        game_state = torch.zeros((num_inits, 3), dtype=torch.float32, device=self.device)
        
        # Initialize the fruits reached bottom count with 0 for each game (no fruits have reached bottom yet)
        # game_state[:, 2] = 0.0  # Already 0 from torch.zeros, but explicit for clarity
        
        return result, game_state
    
    def _reward(self, game_state: torch.Tensor, prev_game_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advanced reward algorithm to evaluate good fruit-catching policy.
        
        DESIGN PHILOSOPHY:
        This reward function implements a multi-faceted evaluation system that goes beyond
        simple score tracking to encourage sophisticated fruit-catching behavior:
        
        1. IMMEDIATE FEEDBACK: Strong immediate rewards/penalties for catching/missing fruits
        2. EFFICIENCY FOCUS: Exponential bonuses for maintaining high catch rates
        3. PROGRESSIVE CHALLENGE: Increasing reward values as game difficulty grows
        4. CONSISTENCY REWARDS: Bonuses for sustained good performance over time
        5. ACTIVITY INCENTIVES: Rewards for engaging with fruits (prevents passive play)
        6. MOMENTUM BUILDING: Extra rewards for consecutive successful actions
        7. ADAPTIVE RESPONSE: Special bonuses for handling high-difficulty situations
        8. EFFICIENCY PRESSURE: Gentle penalties for taking too long per fruit
        
        This creates a reward landscape that encourages the agent to develop:
        - Quick decision-making and precise positioning
        - Consistent performance rather than lucky streaks
        - Adaptive strategies for varying fruit patterns
        - Efficient movement patterns that minimize wasted actions
        
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
        
        # Calculate incremental changes if previous state is provided
        if prev_game_state is not None:
            prev_score = prev_game_state[:, :, 0]
            prev_fruits_reached = prev_game_state[:, :, 2]
            score_delta = score - prev_score
            fruits_delta = fruits_reached_bottom - prev_fruits_reached
        else:
            score_delta = torch.zeros_like(score)
            fruits_delta = torch.zeros_like(fruits_reached_bottom)
        
        # ========== REWARD COMPONENTS ==========
        
        # 1. IMMEDIATE PERFORMANCE REWARD (most important)
        # Strong positive reward for catching fruits, penalty for missing
        immediate_reward = score_delta * 10.0  # +10 for catch, -10 for miss
        
        # 2. CATCH EFFICIENCY BONUS 
        # Reward high catch rates with exponential scaling
        catch_rate_bonus = torch.zeros_like(score, device=device)
        fruits_caught = torch.clamp(score, min=0)  # only positive scores count as catches
        fruits_missed = torch.clamp(-score, min=0)  # negative scores are misses
        
        # Calculate catch rate only when fruits have been encountered
        valid_games = fruits_reached_bottom > 0
        catch_rate = torch.zeros_like(score, device=device)
        catch_rate[valid_games] = fruits_caught[valid_games] / fruits_reached_bottom[valid_games]
        
        # Exponential bonus: 50% = 1.2x, 70% = 2x, 90% = 5x, 100% = 10x bonus
        exp_factor = torch.tensor(2.5, device=device)
        catch_rate_bonus[valid_games] = 5.0 * (torch.exp(exp_factor * catch_rate[valid_games]) - 1) / (torch.exp(exp_factor) - 1)
        
        # 3. PROGRESSIVE DIFFICULTY SCALING
        # Increase reward value as game progresses (later catches are worth more)
        difficulty_multiplier = 1.0 + 0.02 * step_count  # 1.0 → 1.4 over 20 steps → 2.0 over 50 steps
        progressive_bonus = score * difficulty_multiplier * 0.1
        
        # 4. CONSISTENCY REWARD
        # Bonus for maintaining positive performance over time
        consistency_bonus = torch.zeros_like(score, device=device)
        good_performance = score > 0
        sustained_play = step_count > 10
        consistent_performers = good_performance & sustained_play
        
        # Reward: (average score per step) * (consistency factor)
        avg_score_per_step = score / torch.clamp(step_count, min=1)
        consistency_factor = torch.sigmoid(step_count / 10.0)  # 0.5 at step 0, 0.88 at step 20
        consistency_bonus[consistent_performers] = (
            avg_score_per_step[consistent_performers] * 
            consistency_factor[consistent_performers] * 2.0
        )
        
        # 5. ACTIVITY INCENTIVE
        # Small reward for engaging with fruits (prevents passive strategies)
        engagement_reward = 0.5 * fruits_reached_bottom
        
        # 6. EFFICIENCY PRESSURE
        # Gentle penalty for taking too many steps without progress
        steps_per_fruit = torch.zeros_like(score, device=device)
        steps_per_fruit[fruits_reached_bottom > 0] = step_count[fruits_reached_bottom > 0] / fruits_reached_bottom[fruits_reached_bottom > 0]
        
        # Penalty grows when taking more than 5 steps per fruit on average
        efficiency_penalty = torch.zeros_like(score, device=device)
        inefficient = steps_per_fruit > 5.0
        efficiency_penalty[inefficient] = -0.1 * (steps_per_fruit[inefficient] - 5.0)
        
        # 7. MOMENTUM BONUS
        # Extra reward for consecutive successful catches
        momentum_bonus = torch.zeros_like(score, device=device)
        if prev_game_state is not None:
            # Bonus when both this step and cumulative performance are positive
            recent_success = score_delta > 0
            overall_success = score > 0
            momentum_conditions = recent_success & overall_success
            momentum_bonus[momentum_conditions] = 2.0 * score_delta[momentum_conditions]
        
        # 8. ADAPTIVE CHALLENGE RESPONSE
        # Higher rewards when successfully handling multiple fruits
        challenge_bonus = torch.zeros_like(score, device=device)
        high_activity = fruits_delta > 1  # Multiple fruits reached bottom in this step
        successful_handling = score_delta >= 0  # Didn't lose points despite challenge
        challenge_bonus[high_activity & successful_handling] = 3.0
        
        # ========== COMBINE ALL COMPONENTS ==========
        
        total_reward = (
            immediate_reward +          # Primary: immediate catch/miss feedback
            catch_rate_bonus +          # Efficiency: exponential bonus for high success rate
            progressive_bonus +         # Progression: increasing value over time
            consistency_bonus +         # Sustainability: reward for sustained good play
            engagement_reward +         # Activity: encourage fruit interaction
            efficiency_penalty +        # Pressure: avoid dawdling
            momentum_bonus +            # Streaks: reward consecutive successes
            challenge_bonus             # Adaptation: bonus for handling difficulty spikes
        )
        
        # ========== NORMALIZATION AND BOUNDS ==========
        
        # Apply reasonable bounds to prevent training instability
        # Allow negative rewards for bad play, but cap extreme values
        bounded_reward = torch.clamp(total_reward, min=-25.0, max=50.0)
        
        # Scale final reward for stable learning (typical RL rewards are often in [-1, 1] or [-10, 10])
        final_reward = bounded_reward * 0.2
        
        return final_reward, score

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
    
    def _policy_loss(self, inputs :torch.Tensor, actions :torch.Tensor, reward :torch.Tensor):
        log_probs = self.brain(inputs)  # Shape: (batch_size, output_size)

        # This code implements batch-wise action probability extractiona crucial operation in 
        # reinforcement learning that efficiently retrieves the log probabilities of 
        # the specific actions that were actually taken by each agent in a batch. This pattern is 
        # fundamental for policy gradient algorithms where you need to compute gradients 
        # with respect to the probabilities of chosen actions, not all possible actions.
        #
        # The first line `batch_ix = torch.arange(actions.shape[0], device=device)` creates a 
        # batch index tensor containing sequential integers `[0, 1, 2, ..., batch_size-1]`. 
        # The `torch.arange()` function generates this sequence automatically based on 
        # the batch size (first dimension of the actions tensor). The crucial `device=device` 
        # parameter ensures the index tensor is created on the same device (CPU or GPU) as 
        # the other tensors, preventing costly device transfer operations and potential 
        # runtime errors.
        #
        # The advanced indexing operation `log_probs[batch_ix, actions]` uses these batch indices 
        # to perform element-wise selection across the batch dimension. If `log_probs` has shape 
        # `(batch_size, num_actions)` containing log probabilities for all possible actions, 
        # and `actions`  has shape `(batch_size,)` containing the index of the chosen action 
        # for each sample, this indexing extracts exactly one probability per batch element. 
        # For example, if batch element 0 chose action 2 and batch element 1 chose action 5, 
        # the result would contain `[log_probs[0, 2], log_probs[1, 5], ...]`.
        #
        # Why this matters in reinforcement learning: Policy gradient algorithms like REINFORCE or 
        # PPO need to compute gradients that increase the probability of good actions and decrease 
        # the probability of bad actions. However, you only want to update the probabilities of actions 
        # that were actually taken - updating probabilities of actions that weren't chosen would be 
        # meaningless and computationally wasteful. This indexing pattern efficiently extracts 
        # exactly the log probabilities needed for the policy gradient computation `∇log π(a|s)`, 
        # which forms the core of the policy update rule. The comment emphasizes the correctness of 
        # this "batch-wise indexing" because incorrect indexing here would corrupt the entire 
        # learning process by associating wrong action probabilities with wrong experiences.
        #
        # Performance insight: This operation is much more efficient than using loops or 
        # other selection methods, especially on GPU where the parallel indexing can be 
        # performed simultaneously across all batch elements in a single operation.
        batch_ix = torch.arange(actions.shape[0], device=self.device)
        # (num_inits * batch_size, output_size) -> (num_inits * batch_size)
        # Extract the log probabilities of the actions taken in the batch
        log_action_probs = log_probs[batch_ix, actions]  # Correct batch-wise indexing to get the log prob of element i in index i

        return -torch.mean(log_action_probs * reward)  # Scalar loss averaged over the batch
    
    def _train_epoch(self) -> float:

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
            my_return = f_reward_history[-1]  # the reward is the reward at the FINAL time only

            loss = self._policy_loss(f_input_history[t], f_action_history[t], my_return)
            total_loss += loss

        # Compute gradients and update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return last_reward.item(), torch.mean(score_history[-1], dim=-1).item()
    
    def train(self) -> torch.Tensor:
        final_reward = np.zeros(self.config.total_epochs) 
        for epoch in tqdm(range(self.config.total_epochs)):
            final_reward[epoch], score = self._train_epoch()
            print(f"{epoch=}, Avg:{np.mean(final_reward[-1]):.3f}, Score: {score:.3f}")
        
        return final_reward

            