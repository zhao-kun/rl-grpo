#!/usr/bin/env python3
"""
Simple example to run the AI-controlled Fruits Catcher game
Usage: python3 run_game_demo.py [options]
"""

import torch
import argparse
import os
from grpo_fruits_catcher import GameConfig, TrainerConfig
from game_inference import GameInference

def parse_args():
    """Parse command line arguments for the game demo"""
    parser = argparse.ArgumentParser(description='🍎 AI Fruits Catcher Game Demo')
    
    # Model configuration
    model_group = parser.add_argument_group('🤖 Model Configuration')
    model_group.add_argument('--model', '-m', type=str, default=None,
                           help='📂 Specify model file path (e.g., grpo_fruits_catcher-002000.pth)')
    model_group.add_argument('--model-name', type=str, default='grpo_fruits_catcher',
                           help='📁 Model name prefix to search for (default: grpo_fruits_catcher)')
    model_group.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
                           help='💻 Device to run inference on (default: auto)')
    
    # Game configuration overrides
    game_group = parser.add_argument_group('🎮 Game Configuration Overrides')
    game_group.add_argument('--min-interval-steps', type=int, default=None,
                          help='⏱️ Override minimum steps between fruit spawns (overrides model config)')
    
    # Display configuration
    display_group = parser.add_argument_group('📊 Display Configuration')
    display_group.add_argument('--verbose', '-v', action='store_true',
                             help='📋 Show detailed model and game configuration')
    display_group.add_argument('--config-only', action='store_true',
                             help='📄 Only display configuration without running the game')
    display_group.add_argument('--quiet', '-q', action='store_true',
                             help='🔇 Minimal output (no instructions or verbose info)')
    
    return parser.parse_args()

def find_model_file(model_path=None, model_name='grpo_fruits_catcher'):
    """Find the best available model file"""
    if model_path:
        # User specified exact model path
        if os.path.exists(model_path):
            return model_path
        else:
            print(f"❌ Error: Specified model '{model_path}' not found!")
            return None
    
    # Search for models with the given name prefix
    model_candidates = []
    
    # Look for models in current directory
    for file in os.listdir('.'):
        if file.startswith(model_name) and file.endswith('.pth'):
            model_candidates.append(file)
    
    # Sort by epoch number (highest first)
    model_candidates.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]), reverse=True)
    
    # If no custom models found, try default candidates
    if not model_candidates:
        default_candidates = [
            f"{model_name}-003000.pth",
            f"{model_name}-002500.pth", 
            f"{model_name}-002000.pth", 
            f"{model_name}-001000.pth",
            f"{model_name}-000001.pth"
        ]
        for candidate in default_candidates:
            if os.path.exists(candidate):
                return candidate
    elif model_candidates:
        return model_candidates[0]  # Return the highest epoch model
    
    return None

def display_verbose_config(game, args):
    """Display detailed configuration information"""
    print(f"\n{'='*60}")
    print(f"🎮 {'GAME CONFIGURATION':^50} 🎮")
    print(f"{'='*60}")
    
    # Game Configuration
    gc = game.game_config
    print(f"📏 Screen Dimensions: {gc.screen_width} × {gc.screen_height}")
    print(f"🤖 AI Sprite Size: {gc.sprite_width} × {gc.sprite_height}")
    print(f"🍎 Fruits Range: {gc.min_fruits_on_screen} - {gc.max_fruits_on_screen} on screen")
    
    # Show override indicator if min_interval_step_fruits was overridden
    if args.min_interval_steps is not None:
        print(f"⏱️  Fruit Spawn Interval: {gc.min_interval_step_fruits} steps minimum ⚙️ (OVERRIDDEN)")
    else:
        print(f"⏱️  Fruit Spawn Interval: {gc.min_interval_step_fruits} steps minimum")
        
    print(f"📐 View Scaling: {gc.view_width_multiplier} × {gc.view_height_multiplier}")
    print(f"🔄 Refresh Rate: {gc.refresh_timer} ms")
    print(f"🎯 Score Thresholds: Win={gc.win_ended_game_score}, Lose={gc.fail_ended_game_score}")
    
    print(f"\n{'='*60}")
    print(f"🧠 {'TRAINING CONFIGURATION':^50} 🧠")
    print(f"{'='*60}")
    
    # Training Configuration  
    tc = game.trainer_config
    print(f"🔄 Total Epochs: {tc.total_epochs}")
    print(f"📦 Batch Size: {tc.batch_size}")
    print(f"🧠 Hidden Layer Size: {tc.hidden_size}")
    print(f"📈 Learning Rate: {tc.lr_rate}")
    print(f"⏱️  Max Steps per Episode: {tc.max_steps}")
    print(f"🛑 Early Stopping Patience: {tc.patience}")
    print(f"⚡ PyTorch Compile: {'Enabled' if tc.compile else 'Disabled'}")
    
    # Model Architecture Info
    total_params = sum(p.numel() for p in game.gb.parameters())
    trainable_params = sum(p.numel() for p in game.gb.parameters() if p.requires_grad)
    print(f"🏗️  Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Input/Output dimensions
    input_size = gc.get_inputsize()
    print(f"📥 Input Dimensions: {input_size}")
    print(f"📤 Output Actions: {gc.screen_width} (sprite positions)")
    
    print(f"{'='*60}")

def main():
    args = parse_args()
    
    if not args.quiet:
        print("🎮 === AI Fruits Catcher Game Demo === 🍎")
    
    # Find model file
    model_path = find_model_file(args.model, args.model_name)
    
    if model_path is None:
        print("❌ Error: No trained model found!")
        print("📋 Please ensure you have trained a model first.")
        print("▶️  Run: python3 grpo_fruits_catcher.py")
        return
    
    # Determine computation device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if not args.quiet:
        print(f"💻 Using device: {device}")
        print(f"📂 Loading model: {model_path}")
    
    try:
        # Validate override parameters
        if args.min_interval_steps is not None and args.min_interval_steps < 0:
            print(f"❌ Error: --min-interval-steps must be non-negative (got {args.min_interval_steps})")
            return
        
        # Create and run the game
        game = GameInference.from_pretrained(model_path, device, min_interval_step_fruits=args.min_interval_steps)
        
        # Display override information if not quiet
        if not args.quiet and args.min_interval_steps is not None:
            print(f"⚙️  Configuration Override: Fruit spawn interval set to {args.min_interval_steps} steps")
        
        # Display verbose configuration if requested
        if args.verbose:
            display_verbose_config(game, args)
        
        # If config-only mode, don't run the game
        if args.config_only:
            if not args.verbose:
                print("🔧 Use --verbose/-v to see detailed configuration")
            return
        
        if not args.quiet:
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
