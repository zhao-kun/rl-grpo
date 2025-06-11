#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from grpo_fruits_catcher import Trainer, GameConfig, TrainerConfig

def debug_grpo():
    # Create a small test setup
    game_config = GameConfig(screen_width=10, screen_height=6, max_fruits_on_screen=2, min_fruits_on_screen=1)
    trainer_config = TrainerConfig(
        batch_size=2, 
        total_epochs=1, 
        max_steps=10,  # Very short for debugging
        game_config=game_config
    )
    
    trainer = Trainer(trainer_config, 'cpu')
    
    # Generate a single trajectory
    input_history, action_history, score_history, reward_history = trainer._create_trajector()
    
    print("=== DEBUGGING GRPO ===")
    print(f"Reward history shape: {reward_history.shape}")
    print(f"Raw rewards (first 5 steps):")
    for t in range(min(5, reward_history.shape[0])):
        print(f"  Step {t}: {reward_history[t]}")
    
    # Test the GRPO normalization
    batch_size, num_inits = reward_history.shape[1], reward_history.shape[2]
    
    # Compute returns like in the real code
    returns = torch.zeros_like(reward_history)
    running_return = torch.zeros(batch_size, num_inits)
    
    for t in reversed(range(reward_history.shape[0])):
        running_return += reward_history[t]
        returns[t] = running_return.clone()
    
    print(f"\nReturns (first 5 steps):")
    for t in range(min(5, returns.shape[0])):
        print(f"  Step {t}: {returns[t]}")
    
    # Apply GRPO normalization
    epsilon = 1e-7
    group_normed_returns = torch.zeros_like(returns)
    
    for t in range(returns.shape[0]):
        for b in range(batch_size):
            group_returns = returns[t, b, :]
            mean_return = torch.mean(group_returns)
            std_return = torch.std(group_returns) + epsilon
            print(f"Step {t}, Batch {b}: mean={mean_return:.4f}, std={std_return:.4f}")
            group_normed_returns[t, b, :] = (group_returns - mean_return) / std_return
    
    print(f"\nGroup-normalized returns (first 5 steps):")
    for t in range(min(5, group_normed_returns.shape[0])):
        print(f"  Step {t}: {group_normed_returns[t]}")
    
    # Check if the brain outputs reasonable logits
    sample_input = input_history[0].reshape(-1, trainer_config.game_config.get_inputsize())[:4]  # Take first 4 samples
    with torch.no_grad():
        logits = trainer.brain(sample_input)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        
    print(f"\nBrain outputs (first 4 samples):")
    print(f"Logits: {logits}")
    print(f"Log probs: {log_probs}")
    print(f"Probabilities: {probs}")

if __name__ == "__main__":
    debug_grpo()
