#!/usr/bin/env python3
"""
Demonstration of the advanced reward algorithm for the fruit catcher game.

This script shows how the reward function evaluates different game scenarios
and encourages good fruit-catching policies.
"""

import torch
from grpo_fruits_catcher import GameConfig, TrainerConfig, Trainer

def demo_reward_algorithm():
    """Demonstrate the reward algorithm with various game scenarios."""
    
    print("🍎 Fruit Catcher Reward Algorithm Demonstration")
    print("=" * 60)
    
    # Setup
    device = "cpu"
    game_config = GameConfig(screen_width=20, screen_height=11, max_fruits_on_screen=3)
    train_config = TrainerConfig(game_config=game_config)
    trainer = Trainer(train_config, device)
    
    # Test scenarios: [score, step_count, fruits_reached_bottom]
    scenarios = [
        # Perfect performance scenarios
        (torch.tensor([[[5.0, 10.0, 5.0]]]), "Perfect catch rate (5/5 fruits caught)"),
        (torch.tensor([[[10.0, 20.0, 10.0]]]), "Excellent sustained performance"),
        
        # Good performance scenarios  
        (torch.tensor([[[3.0, 10.0, 5.0]]]), "Good catch rate (3/5 fruits caught)"),
        (torch.tensor([[[8.0, 30.0, 10.0]]]), "Consistent good play over time"),
        
        # Average performance scenarios
        (torch.tensor([[[0.0, 10.0, 5.0]]]), "Break-even performance (equal catches/misses)"),
        (torch.tensor([[[2.0, 15.0, 8.0]]]), "Slightly positive with some misses"),
        
        # Poor performance scenarios
        (torch.tensor([[[-3.0, 10.0, 5.0]]]), "Poor catch rate (more misses than catches)"),
        (torch.tensor([[[-5.0, 20.0, 8.0]]]), "Consistently missing fruits"),
        
        # Special scenarios
        (torch.tensor([[[1.0, 50.0, 2.0]]]), "Slow but cautious play"),
        (torch.tensor([[[15.0, 25.0, 20.0]]]), "High activity, good performance"),
    ]
    
    print("\n📊 Reward Analysis for Different Game Scenarios")
    print("-" * 60)
    
    for i, (game_state, description) in enumerate(scenarios):
        # Calculate rewards
        reward, score = trainer._reward(game_state)
        
        # Extract game stats
        current_score = game_state[0, 0, 0].item()
        steps = game_state[0, 0, 1].item()
        fruits_total = game_state[0, 0, 2].item()
        
        # Calculate derived metrics
        catch_rate = max(0, current_score) / fruits_total if fruits_total > 0 else 0
        efficiency = fruits_total / steps if steps > 0 else 0
        
        print(f"\n{i+1:2d}. {description}")
        print(f"    Game State: Score={current_score:+.1f}, Steps={steps:.0f}, Fruits={fruits_total:.0f}")
        print(f"    Metrics: Catch Rate={catch_rate:.1%}, Efficiency={efficiency:.2f} fruits/step")
        print(f"    💰 REWARD: {reward.item():+.2f}")
    
    print("\n" + "=" * 60)
    print("🎯 Reward Algorithm Key Features:")
    print("   • Immediate feedback: ±10 points per catch/miss")
    print("   • Efficiency bonus: Exponential rewards for high catch rates")
    print("   • Progressive scaling: Later game performance worth more")
    print("   • Consistency rewards: Sustained good play gets bonuses")
    print("   • Activity incentives: Engagement prevents passive strategies")
    print("   • Momentum bonuses: Consecutive successes get extra rewards")
    print("   • Adaptive response: Handling difficult situations gets bonuses")
    print("   • Efficiency pressure: Penalties for dawdling")

def demo_incremental_rewards():
    """Demonstrate how incremental reward calculation works."""
    
    print("\n\n🔄 Incremental Reward Calculation Demo")
    print("=" * 60)
    
    device = "cpu"
    game_config = GameConfig()
    train_config = TrainerConfig(game_config=game_config)
    trainer = Trainer(train_config, device)
    
    # Simulate a sequence of game states
    states = [
        torch.tensor([[[0.0, 1.0, 0.0]]]),  # Initial state
        torch.tensor([[[1.0, 2.0, 1.0]]]),  # Caught a fruit
        torch.tensor([[[1.0, 3.0, 2.0]]]),  # Missed a fruit  
        torch.tensor([[[2.0, 4.0, 3.0]]]),  # Caught another fruit
        torch.tensor([[[4.0, 5.0, 5.0]]]),  # Caught 2 fruits in one step!
    ]
    
    descriptions = [
        "Initial state (no fruits yet)",
        "First fruit caught! (+1 score)",
        "Second fruit missed (+0 score, but fruit reached bottom)",
        "Third fruit caught! (+1 score)",
        "Double catch! (+2 score from 2 fruits)"
    ]
    
    print("\nStep-by-step reward calculation:")
    print("-" * 40)
    
    prev_state = None
    for step, (state, desc) in enumerate(zip(states, descriptions)):
        reward, score = trainer._reward(state, prev_state)
        
        score_val = state[0, 0, 0].item()
        steps_val = state[0, 0, 1].item()
        fruits_val = state[0, 0, 2].item()
        
        print(f"\nStep {step + 1}: {desc}")
        print(f"  State: Score={score_val:+.1f}, Steps={steps_val:.0f}, Fruits={fruits_val:.0f}")
        print(f"  Reward: {reward.item():+.2f}")
        
        if step > 0:
            prev_score = prev_state[0, 0, 0].item()
            score_change = score_val - prev_score
            print(f"  Score Change: {score_change:+.1f} → Base reward: {score_change * 10:+.1f}")
        
        prev_state = state.clone()

if __name__ == "__main__":
    demo_reward_algorithm()
    demo_incremental_rewards()
    
    print("\n\n✅ Reward algorithm demonstration complete!")
    print("The algorithm successfully evaluates policies based on:")
    print("  - Immediate performance (catches vs misses)")
    print("  - Long-term efficiency and consistency") 
    print("  - Adaptive responses to varying difficulty")
    print("  - Balanced incentives for active, skillful play")
