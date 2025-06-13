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
    model_path = "grpo_fruits_catcher-000001.pth"
    
    # Determine computation device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Create and run the game
        game = GameInference.from_pretrained(model_path, device)
        
        print("\n🎮 Game Instructions:")
        print("• Red circles = Falling fruits (catch them!)")
        print("• Green rectangle = AI-controlled sprite")
        print("• Score increases when catching fruits")
        print("• Score decreases when missing fruits")
        print("• Game ends when score reaches", game.game_config.ended_game_score)
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
