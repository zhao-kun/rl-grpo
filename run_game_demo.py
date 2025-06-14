#!/usr/bin/env python3
"""
Simple example to run the AI-controlled Fruits Catcher game
Usage: python3 run_game_demo.py
"""

import torch
from grpo_fruits_catcher import GameConfig, TrainerConfig
from game_inference import GameInference

def main():
    print("🎮 === AI Fruits Catcher Game Demo === 🍎")
    
    # Use the existing trained model (try multiple options)
    model_candidates = ["grpo_fruits_catcher-003000.pth", "grpo_fruits_catcher-002000.pth", "grpo_fruits_catcher-000001.pth"]
    model_path = None
    
    for candidate in model_candidates:
        import os
        if os.path.exists(candidate):
            model_path = candidate
            break
    
    if model_path is None:
        print("❌ Error: No trained model found!")
        print("📋 Please ensure you have trained a model first.")
        print("▶️  Run: python3 grpo_fruits_catcher.py")
        return
    
    # Determine computation device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Using device: {device}")
    print(f"📂 Loading model: {model_path}")
    
    try:
        # Create and run the game
        game = GameInference.from_pretrained(model_path, device)
        
        print("\n🎮 Game Instructions:")
        print("• 🍎 Fruit emojis = Falling fruits (catch them!)")
        print("• 🤖 Green rectangle with eyes = AI-controlled sprite")
        print("• 📈 Score increases when catching fruits")
        print("• 📉 Score decreases when missing fruits")
        print(f"• 🏆 AI WINS when score reaches {game.game_config.win_ended_game_score}")
        print(f"• 💥 AI LOSES when score drops to {game.game_config.fail_ended_game_score}")
        print(f"• 🎓 Model was trained for {game.trainer_config.total_epochs} epochs")
        print(f"• 🧠 Model hidden size: {game.trainer_config.hidden_size}")
        print("• ⌨️  Press ESC to quit anytime")
        print("\n⏳ Starting game in 1 seconds...")
        
        import time
        time.sleep(1)
        
        game.run()
        
    except FileNotFoundError:
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("📋 Please ensure you have trained a model first.")
        print("▶️  Run: python3 grpo_fruits_catcher.py")
        
    except KeyboardInterrupt:
        print("\n⏹️  Game interrupted by user")
        
    except Exception as e:
        print(f"❌ Error running game: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
