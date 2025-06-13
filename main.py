import torch
from grpo_fruits_catcher import Trainer, GameConfig, TrainerConfig


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    torch.set_float32_matmul_precision('high')
    # Define the game configuration
    game_config = GameConfig()

    # Define the trainer configuration with optimized hyperparameters
    trainer_config = TrainerConfig(
        game_config=game_config, 
        total_epochs=2000,  # Shorter training to see faster results
        lr_rate=1e-4,  # Reduced learning rate for stable training
        batch_size=16,  # Good batch size for stability
        max_steps=150,  # Optimal episode length
        hidden_size=512,  # Larger network for better learning capacity
        compile=False
    )

    # Create a trainer instance
    trainer = Trainer(trainer_config, device)

    # Start training
    trainer.train()
    trainer.save("grpo_fruits_catcher")

    
if __name__ == "__main__":
    main()
