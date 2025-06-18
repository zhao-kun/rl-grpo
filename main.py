import torch
import argparse
from grpo_fruits_catcher import Trainer, GameConfig, TrainerConfig


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def parse_args():
    """Parse command line arguments for training configuration"""
    parser = argparse.ArgumentParser(description='🍎 Fruits Catcher GRPO Training')
    
    # 🎮 Game Configuration Arguments
    game_group = parser.add_argument_group('🎮 Game Configuration')
    game_group.add_argument('--screen-width', type=int, default=20,
                           help='🔢 Game screen width (default: 20)')
    game_group.add_argument('--screen-height', type=int, default=15,
                           help='🔢 Game screen height (default: 15)')
    game_group.add_argument('--sprite-width', type=int, default=3,
                           help='🤖 AI sprite width (default: 3)')
    game_group.add_argument('--sprite-height', type=int, default=1,
                           help='🤖 AI sprite height (default: 1)')
    game_group.add_argument('--max-fruits', type=int, default=3,
                           help='🍎 Maximum fruits on screen (default: 3)')
    game_group.add_argument('--min-fruits', type=int, default=1,
                           help='🍎 Minimum fruits on screen (default: 1)')
    game_group.add_argument('--min-interval-steps', type=int, default=5,
                           help='⏱️ Minimum steps between fruit spawns (default: 5)')
    game_group.add_argument('--view-height-multiplier', type=float, default=50.0,
                           help='📐 View height scaling factor (default: 50.0)')
    game_group.add_argument('--view-width-multiplier', type=float, default=50.0,
                           help='📐 View width scaling factor (default: 50.0)')
    game_group.add_argument('--refresh-timer', type=int, default=150,
                           help='🔄 Game refresh timer in ms (default: 150)')
    game_group.add_argument('--fail-score', type=int, default=-30,
                           help='💥 Score threshold for game failure (default: -30)')
    game_group.add_argument('--win-score', type=int, default=30,
                           help='🏆 Score threshold for game victory (default: 30)')
    
    # 🧠 Training Configuration Arguments
    training_group = parser.add_argument_group('🧠 Training Configuration')
    training_group.add_argument('--hidden-size', type=int, default=2048,
                               help='🧠 Neural network hidden layer size (default: 2048)')
    training_group.add_argument('--batch-size', type=int, default=32,
                               help='📦 Training batch size (default: 32)')
    training_group.add_argument('--total-epochs', type=int, default=2000,
                               help='🔄 Total training epochs (default: 2000)')
    training_group.add_argument('--max-steps', type=int, default=100,
                               help='⏱️ Maximum steps per episode (default: 100)')
    training_group.add_argument('--lr-rate', type=float, default=1e-4,
                               help='📈 Learning rate (default: 1e-4)')
    training_group.add_argument('--patience', type=int, default=500,
                               help='🛑 Early stopping patience in epochs (default: 500)')
    training_group.add_argument('--compile', action='store_true',
                               help='⚡ Enable torch.compile for faster training')
    training_group.add_argument('--no-compile', action='store_true',
                               help='🐌 Disable torch.compile (default)')
    training_group.add_argument('--save-checkpoint-per-num-epoch', type=int, default=200,
                               help='💾 Save checkpoint every N epochs (default: 200)')
    training_group.add_argument('--save-best-model', action='store_true',
                               help='🏆 Save the best model during training (default)')

    # 💾 Output Configuration
    output_group = parser.add_argument_group('💾 Output Configuration')
    output_group.add_argument('--model-name', type=str, default='grpo_fruits_catcher',
                             help='📂 Model save name (default: grpo_fruits_catcher)')
    output_group.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'],
                             default='auto', help='💻 Training device (default: auto)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"💻 Using device: {device}")
    
    torch.set_float32_matmul_precision('high')
    
    # Define the game configuration from arguments
    game_config = GameConfig(
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        sprite_width=args.sprite_width,
        sprite_height=args.sprite_height,
        max_fruits_on_screen=args.max_fruits,
        min_fruits_on_screen=args.min_fruits,
        min_interval_step_fruits=args.min_interval_steps,
        view_height_multiplier=args.view_height_multiplier,
        view_width_multiplier=args.view_width_multiplier,
        refresh_timer=args.refresh_timer,
        fail_ended_game_score=args.fail_score,
        win_ended_game_score=args.win_score
    )

    # Handle compile argument (--no-compile takes precedence)
    compile_model = args.compile and not args.no_compile

    # Define the trainer configuration from arguments
    trainer_config = TrainerConfig(
        game_config=game_config,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        total_epochs=args.total_epochs,
        max_steps=args.max_steps,
        lr_rate=args.lr_rate,
        compile=compile_model,
        patience=args.patience,
        save_checkpoint_per_num_epoch=args.save_checkpoint_per_num_epoch,
        save_best_model=args.save_best_model,
        model_name=args.model_name
    )

    print(f"🎮 Game Configuration:")
    print(f"  📏 Screen: {game_config.screen_width}×{game_config.screen_height}")
    print(f"  🤖 Sprite: {game_config.sprite_width}×{game_config.sprite_height}")
    print(f"  🍎 Fruits: {game_config.min_fruits_on_screen}-{game_config.max_fruits_on_screen}")
    print(f"  🎯 Scores: Win={game_config.win_ended_game_score}, Fail={game_config.fail_ended_game_score}")
    
    print(f"\n🧠 Training Configuration:")
    print(f"  🔄 Epochs: {trainer_config.total_epochs}")
    print(f"  📦 Batch Size: {trainer_config.batch_size}")
    print(f"  🧠 Hidden Size: {trainer_config.hidden_size}")
    print(f"  📈 Learning Rate: {trainer_config.lr_rate}")
    print(f"  ⏱️ Max Steps: {trainer_config.max_steps}")
    print(f"  🛑 Early Stopping Patience: {trainer_config.patience}")
    print(f"  ⚡ Compile: {'Yes' if trainer_config.compile else 'No'}")
    
    # Create a trainer instance
    trainer = Trainer(trainer_config, device)

    # Start training
    print(f"\n🚀 Starting training...")
    trainer.train()
    
    # Save the model
    print(f"\n💾 Saving model as '{trainer_config.model_name}'...")
    trainer.save(trainer_config.model_name)
    print(f"✅ Training completed successfully!")

    
if __name__ == "__main__":
    main()
