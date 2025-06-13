#!/usr/bin/env python3
"""
Simple example to run the AI-controlled Fruits Catcher game
Usage: python3 run_game_demo.py
"""

import torch
from grpo_fruits_catcher import GameConfig, TrainerConfig
from game_inference import GameInference

def main():
    print("=== AI Fruits Catcher Game Demo ===")
    
    # Use the existing trained model
    model_path = "grpo_fruits_catcher-002000.pth"
    
    # Game configuration optimized for visual demonstration
    game_config = GameConfig(
        screen_width=20,          # Game grid width
        screen_height=11,         # Game grid height
        sprite_width=3,           # Player sprite width
        sprite_height=1,          # Player sprite height
        max_fruits_on_screen=3,   # Maximum fruits simultaneously
        min_fruits_on_screen=1,   # Minimum fruits to maintain
        min_interval_step_fruits=3, # Minimum spacing between fruits
        view_height_multiplier=50.0,  # Visual scaling factor
        view_width_multiplier=50.0,   # Visual scaling factor
        refresh_timer=150,        # Game update interval (milliseconds)
        ended_game_score=-30      # Game over score threshold
    )
    
    trainer_config = TrainerConfig(
        hidden_size=512,
        game_config=game_config,
        compile=False  # Disable compilation for inference
    )
    
    # Determine computation device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Create and run the game
        game = GameInference(model_path, trainer_config, device)
        
        print("\n🎮 Game Instructions:")
        print("• Red circles = Falling fruits (catch them!)")
        print("• Green rectangle = AI-controlled sprite")
        print("• Score increases when catching fruits")
        print("• Score decreases when missing fruits")
        print("• Game ends when score reaches", game_config.ended_game_score)
        print("• Press ESC to quit anytime")
        print("\nStarting game in 3 seconds...")
        
        import time
        time.sleep(3)
        
        game.run()
        
    except FileNotFoundError:
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("Please ensure you have trained a model first.")
        print("Run: python3 grpo_fruits_catcher.py")
        
    except KeyboardInterrupt:
        print("\n⏹️  Game interrupted by user")
        
    except Exception as e:
        print(f"❌ Error running game: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
