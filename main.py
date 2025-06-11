from grpo_fruits_catcher import Trainer, GameConfig, TrainerConfig



def main():
    # Define the game configuration
    game_config = GameConfig()

    # Define the trainer configuration
    trainer_config = TrainerConfig(game_config=game_config)

    # Create a trainer instance
    trainer = Trainer(trainer_config, 'cuda:0')

    # Start training
    trainer.train()

    
if __name__ == "__main__":
    main()