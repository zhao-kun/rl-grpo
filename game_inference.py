import torch
import pygame
from typing import Tuple
from grpo_fruits_catcher import GameConfig, Trainer, TrainerConfig, GameBrain, GameEngine


class GameInference:
    device: str ='cpu'
    game_config: GameConfig = None
    trainer_config: TrainerConfig = None
    gb : GameBrain = None
    engine: GameEngine = None
    
    def __init__(self, model_path: str, trainer_config: TrainerConfig = None, device: str = 'cpu'):
        assert model_path is not None, "Model path must be provided"
        assert trainer_config is not None, "TrainerConfig must be provided"
        self.device = device
        self.model_path = model_path
        self.trainer_config = trainer_config
        self.game_config = trainer_config.game_config
        self.gb = GameBrain.from_pretrained(model_path, trainer_config).to(device)
        self.engine = GameEngine(trainer_config, self.gb)
    
    
    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Fruits Catcher - AI Inference")
        screen_width = int(self.game_config.screen_width * self.game_config.view_width_multiplier)
        screen_height = int(self.game_config.screen_height * self.game_config.view_height_multiplier)
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        
        # Colors for rendering
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)      # Fruits
        self.GREEN = (0, 255, 0)    # Sprite (player)
        self.BLUE = (0, 0, 255)     # UI elements
        self.GRAY = (128, 128, 128) # Grid lines
    
    def _create_single_game_init(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create initial state for a single game (not batch)"""
        # Create single sprite position (start in middle)
        sprite_pos = torch.tensor([self.game_config.screen_width // 2], dtype=torch.float32, device=self.device)
        
        # Create initial fruits (only first fruit is active)
        max_fruits = self.game_config.max_fruits_on_screen
        fruits = torch.zeros((max_fruits, 3), dtype=torch.float32, device=self.device)
        
        # Set first fruit properties
        import random
        fruits[0, 0] = random.randint(0, self.game_config.screen_width - 1)  # x position
        fruits[0, 1] = 0  # y position (always starts at 0)
        fruits[0, 2] = 1.0  # activation status (active)
        
        # Flatten fruits and concatenate with sprite position
        fruits_flat = fruits.view(-1)
        inputs_state = torch.cat([sprite_pos, fruits_flat]).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, input_size)
        
        # Initialize game state: [score, step_count, fruits_reached_bottom]
        game_state = torch.zeros((1, 1, 3), dtype=torch.float32, device=self.device)
        
        return inputs_state, game_state
    
    def _draw_game(self, inputs_state: torch.Tensor, game_state: torch.Tensor, last_action: int = None):
        """Draw the current game state"""
        self.screen.fill(self.BLACK)
        
        # Extract game data (remove batch dimensions)
        sprite_pos = inputs_state[0, 0, 0].item()
        score = game_state[0, 0, 0].item()
        step_count = int(game_state[0, 0, 1].item())
        fruits_reached_bottom = int(game_state[0, 0, 2].item())
        
        # Extract fruits data
        max_fruits = self.game_config.max_fruits_on_screen
        fruits_data = inputs_state[0, 0, 1:].view(max_fruits, 3)
        
        # Scale factors for visualization
        scale_x = self.game_config.view_width_multiplier
        scale_y = self.game_config.view_height_multiplier
        
        # Draw grid (optional, for better visualization)
        for x in range(self.game_config.screen_width + 1):
            pygame.draw.line(self.screen, self.GRAY, 
                           (x * scale_x, 0), 
                           (x * scale_x, self.game_config.screen_height * scale_y), 1)
        for y in range(self.game_config.screen_height + 1):
            pygame.draw.line(self.screen, self.GRAY, 
                           (0, y * scale_y), 
                           (self.game_config.screen_width * scale_x, y * scale_y), 1)
        
        # Draw fruits
        active_fruits = 0
        for i in range(max_fruits):
            fruit_x, fruit_y, fruit_active = fruits_data[i]
            if fruit_active.item() == 1.0:  # Only draw active fruits
                active_fruits += 1
                # Scale fruit position
                scaled_x = int(fruit_x.item() * scale_x)
                scaled_y = int(fruit_y.item() * scale_y)
                # Draw fruit as a red circle
                pygame.draw.circle(self.screen, self.RED, 
                                 (scaled_x + int(scale_x//2), scaled_y + int(scale_y//2)), 
                                 int(min(scale_x, scale_y) // 3))
        
        # Draw sprite (player)
        sprite_width = self.game_config.sprite_width
        sprite_y = self.game_config.screen_height - 1
        
        # Calculate sprite boundaries
        sprite_left = max(0, sprite_pos - sprite_width // 2)
        sprite_right = min(self.game_config.screen_width - 1, sprite_pos + sprite_width // 2)
        
        # Draw sprite as a green rectangle
        sprite_rect = pygame.Rect(
            int(sprite_left * scale_x),
            int(sprite_y * scale_y),
            int((sprite_right - sprite_left + 1) * scale_x),
            int(scale_y)
        )
        pygame.draw.rect(self.screen, self.GREEN, sprite_rect)
        
        # Draw UI information
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {score:.1f}", True, self.WHITE)
        step_text = font.render(f"Steps: {step_count}", True, self.WHITE)
        fruits_text = font.render(f"Active Fruits: {active_fruits}", True, self.WHITE)
        fruits_bottom_text = font.render(f"Fruits Reached Bottom: {fruits_reached_bottom}", True, self.WHITE)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(step_text, (10, 50))
        self.screen.blit(fruits_text, (10, 90))
        self.screen.blit(fruits_bottom_text, (10, 130))
        
        # Show last action
        if last_action is not None:
            action_names = ["LEFT", "STAY", "RIGHT"]
            action_text = font.render(f"Last Action: {action_names[last_action]}", True, self.BLUE)
            self.screen.blit(action_text, (10, 170))
        
        # Draw sprite position debug info
        sprite_debug_text = pygame.font.Font(None, 24).render(f"Sprite Pos: {sprite_pos:.1f}", True, self.WHITE)
        self.screen.blit(sprite_debug_text, (10, 210))
        
        # Draw instructions
        instruction_font = pygame.font.Font(None, 24)
        instruction_text = instruction_font.render("AI is playing - Press ESC to quit", True, self.WHITE)
        self.screen.blit(instruction_text, (10, self.screen.get_height() - 30))
        
        pygame.display.flip()

    def run(self):
        """Run the game inference with AI control"""
        self._init_pygame()
        
        # Initialize game
        inputs_state, game_state = self._create_single_game_init()
        
        # Game timing
        last_update_time = 0
        running = True
        last_action = None
        
        print("Starting AI-controlled Fruits Catcher game...")
        print(f"Game will end when score reaches {self.game_config.ended_game_score}")
        
        while running:
            current_time = pygame.time.get_ticks()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Update game state at specified intervals
            if current_time - last_update_time >= self.game_config.refresh_timer:
                # Update game using AI brain
                new_inputs_state, actions, new_game_state = self.engine.update(inputs_state, game_state)
                inputs_state = new_inputs_state
                game_state = new_game_state
                last_action = actions[0, 0].item()  # Extract action for single game
                last_update_time = current_time
                
                # Get current score
                current_score = game_state[0, 0, 0].item()
                
                # Check if game should end
                if current_score <= self.game_config.ended_game_score:
                    print(f"Game Over! Final Score: {current_score:.1f}")
                    running = False
                
                # Optional: print periodic updates
                step_count = int(game_state[0, 0, 1].item())
                if step_count % 50 == 0:  # Print every 50 steps
                    print(f"Step {step_count}: Score = {current_score:.1f}")
            
            # Draw the game
            self._draw_game(inputs_state, game_state, last_action)
            
            # Control frame rate
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
        print("Game finished!")