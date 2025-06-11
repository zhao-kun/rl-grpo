import torch
from grpo_fruits_catcher import Trainer, GameConfig, TrainerConfig


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    # Define the game configuration
    game_config = GameConfig()

    # Define the trainer configuration
    trainer_config = TrainerConfig(game_config=game_config, 
                                   total_epochs=600, 
                                   lr_rate=2e-4,
                                   compile=True)

    # Create a trainer instance
    trainer = Trainer(trainer_config, device)

    # Start training
    trainer.train()
    trainer.save("grpo_fruits_catcher")

    
if __name__ == "__main__":
    main()