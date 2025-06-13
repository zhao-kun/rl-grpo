import torch
import pygame
import math
import random
from typing import Tuple, List, Dict
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
        
        # Initialize score tracking for effects
        self.previous_score = 0
    
    
    def _init_pygame(self):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("🍎 Fruits Catcher - AI Master 🤖")
        screen_width = int(self.game_config.screen_width * self.game_config.view_width_multiplier)
        screen_height = int(self.game_config.screen_height * self.game_config.view_height_multiplier)
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        
        # Enhanced color palette
        self.BACKGROUND = (20, 30, 50)      # Dark blue background
        self.WHITE = (255, 255, 255)
        self.GREEN = (50, 200, 50)          # Player color
        self.GOLD = (255, 215, 0)           # Score highlights
        self.CYAN = (0, 255, 255)           # Action indicators
        self.GRAY = (80, 80, 100)           # Grid lines
        self.RED = (255, 80, 80)            # Warning color
        
        # Fruit emojis and colors
        self.fruit_types = [
            {"emoji": "🍎", "color": (255, 100, 100), "name": "Apple"},
            {"emoji": "🍊", "color": (255, 165, 0), "name": "Orange"}, 
            {"emoji": "🍌", "color": (255, 255, 0), "name": "Banana"},
            {"emoji": "🍇", "color": (147, 112, 219), "name": "Grapes"},
            {"emoji": "🍓", "color": (255, 182, 193), "name": "Strawberry"},
            {"emoji": "🥝", "color": (173, 255, 47), "name": "Kiwi"},
        ]
        
        # Initialize fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_emoji = pygame.font.Font(None, 64)  # For fruit emojis
        
        # Particle effects system
        self.particles = []
        self.catch_effects = []
        
        # Animation tracking
        self.last_score = 0
        self.score_pulse = 0
    
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
    
    def _add_catch_particles(self, x, y, fruit_type):
        """Add particle effects when a fruit is caught"""
        for _ in range(15):
            particle = Particle(
                x + random.randint(-20, 20),
                y + random.randint(-10, 10),
                fruit_type['color'],
                random.uniform(-3, 3),
                random.uniform(-8, -2),
                random.randint(30, 60)
            )
            self.particles.append(particle)
        
        # Add catch effect text
        catch_effect = CatchEffect(x, y - 30, fruit_type)
        self.catch_effects.append(catch_effect)
    
    def _update_effects(self):
        """Update all visual effects"""
        # Update particles
        self.particles = [p for p in self.particles if p.update()]
        
        # Update catch effects
        self.catch_effects = [e for e in self.catch_effects if e.update()]
        
        # Update score pulse animation
        if self.score_pulse > 0:
            self.score_pulse -= 2
    
    def _draw_background(self):
        """Draw enhanced background with gradient"""
        # Create gradient background
        for y in range(self.screen.get_height()):
            color_ratio = y / self.screen.get_height()
            r = int(20 + (40 - 20) * color_ratio)
            g = int(30 + (60 - 30) * color_ratio)  
            b = int(50 + (100 - 50) * color_ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.screen.get_width(), y))
        
        # Draw subtle grid
        scale_x = self.game_config.view_width_multiplier
        scale_y = self.game_config.view_height_multiplier
        
        for x in range(0, self.game_config.screen_width + 1, 2):
            pygame.draw.line(self.screen, self.GRAY, 
                           (x * scale_x, 0), 
                           (x * scale_x, self.game_config.screen_height * scale_y), 1)
        for y in range(0, self.game_config.screen_height + 1, 2):
            pygame.draw.line(self.screen, self.GRAY, 
                           (0, y * scale_y), 
                           (self.game_config.screen_width * scale_x, y * scale_y), 1)
    
    def _draw_fruits(self, fruits_data, scale_x, scale_y):
        """Draw fruits with emoji and enhanced effects"""
        active_fruits = 0
        for i in range(self.game_config.max_fruits_on_screen):
            fruit_x, fruit_y, fruit_active = fruits_data[i]
            if fruit_active.item() == 1.0:
                active_fruits += 1
                
                # Choose fruit type based on position (pseudo-random but consistent)
                fruit_type = self.fruit_types[i % len(self.fruit_types)]
                
                scaled_x = int(fruit_x.item() * scale_x)
                scaled_y = int(fruit_y.item() * scale_y)
                center_x = scaled_x + int(scale_x//2)
                center_y = scaled_y + int(scale_y//2)
                
                # Draw fruit shadow
                shadow_offset = 3
                pygame.draw.circle(self.screen, (0, 0, 0, 100), 
                                 (center_x + shadow_offset, center_y + shadow_offset), 
                                 int(min(scale_x, scale_y) // 2))
                
                # Draw fruit glow effect
                glow_radius = int(min(scale_x, scale_y) // 1.5)
                for r in range(glow_radius, 0, -2):
                    alpha = int(30 * (1 - r / glow_radius))
                    glow_color = (*fruit_type['color'], alpha)
                    pygame.draw.circle(self.screen, fruit_type['color'], 
                                     (center_x, center_y), r)
                
                # Draw fruit emoji (larger and more prominent)
                emoji_surface = self.font_emoji.render(fruit_type['emoji'], True, self.WHITE)
                emoji_rect = emoji_surface.get_rect(center=(center_x, center_y))
                self.screen.blit(emoji_surface, emoji_rect)
                
                # Add floating animation
                float_offset = int(3 * math.sin(pygame.time.get_ticks() * 0.01 + i))
                emoji_rect.y += float_offset
                
        return active_fruits
    
    def _draw_player(self, sprite_pos, scale_x, scale_y):
        """Draw enhanced player sprite"""
        sprite_width = self.game_config.sprite_width
        sprite_y = self.game_config.screen_height - 1
        
        sprite_left = max(0, sprite_pos - sprite_width // 2)
        sprite_right = min(self.game_config.screen_width - 1, sprite_pos + sprite_width // 2)
        
        # Player body
        sprite_rect = pygame.Rect(
            int(sprite_left * scale_x),
            int(sprite_y * scale_y),
            int((sprite_right - sprite_left + 1) * scale_x),
            int(scale_y)
        )
        
        # Draw player with glow effect
        glow_rect = sprite_rect.inflate(6, 6)
        pygame.draw.rect(self.screen, (100, 255, 100), glow_rect, 3)
        pygame.draw.rect(self.screen, self.GREEN, sprite_rect)
        
        # Add player "eyes" 🤖
        eye_size = 4
        eye_y = sprite_rect.y + 3
        left_eye_x = sprite_rect.x + sprite_rect.width // 3
        right_eye_x = sprite_rect.x + 2 * sprite_rect.width // 3
        
        pygame.draw.circle(self.screen, self.WHITE, (left_eye_x, eye_y), eye_size)
        pygame.draw.circle(self.screen, self.WHITE, (right_eye_x, eye_y), eye_size)
        pygame.draw.circle(self.screen, (0, 0, 0), (left_eye_x, eye_y), 2)
        pygame.draw.circle(self.screen, (0, 0, 0), (right_eye_x, eye_y), 2)
    
    def _draw_ui(self, score, step_count, active_fruits, fruits_reached_bottom, last_action):
        """Draw enhanced UI with animations"""
        # Check for score increase
        if score > self.last_score:
            self.score_pulse = 20
            self.last_score = score
        
        # Animated score display
        score_color = self.GOLD if self.score_pulse > 0 else self.WHITE
        score_size = self.font_large.get_height() + self.score_pulse
        
        # Create pulsing score text
        pulse_font = pygame.font.Font(None, 48 + self.score_pulse)
        score_text = pulse_font.render(f"🏆 Score: {score:.1f}", True, score_color)
        self.screen.blit(score_text, (20, 20))
        
        # Game stats with icons
        stats_y = 80
        step_text = self.font_medium.render(f"⏱️ Steps: {step_count}", True, self.WHITE)
        fruits_text = self.font_medium.render(f"🍎 Active: {active_fruits}", True, self.WHITE)
        bottom_text = self.font_medium.render(f"💥 Missed: {fruits_reached_bottom}", True, self.RED if fruits_reached_bottom > 0 else self.WHITE)
        
        self.screen.blit(step_text, (20, stats_y))
        self.screen.blit(fruits_text, (20, stats_y + 35))
        self.screen.blit(bottom_text, (20, stats_y + 70))
        
        # Action indicator with animation
        if last_action is not None:
            action_names = ["⬅️ LEFT", "⏸️ STAY", "➡️ RIGHT"]
            action_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
            
            action_text = self.font_medium.render(f"🎮 {action_names[last_action]}", True, action_colors[last_action])
            action_rect = action_text.get_rect()
            action_rect.topright = (self.screen.get_width() - 20, 20)
            
            # Add action background
            bg_rect = action_rect.inflate(10, 5)
            pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect)
            pygame.draw.rect(self.screen, action_colors[last_action], bg_rect, 2)
            
            self.screen.blit(action_text, action_rect)
        
        # AI status indicator
        ai_text = self.font_small.render("🤖 AI PLAYING - Press ESC to quit", True, self.CYAN)
        ai_rect = ai_text.get_rect()
        ai_rect.bottomleft = (20, self.screen.get_height() - 10)
        self.screen.blit(ai_text, ai_rect)
        
        # Performance indicator
        fps = int(self.clock.get_fps())
        fps_color = self.WHITE if fps > 30 else self.RED
        fps_text = self.font_small.render(f"FPS: {fps}", True, fps_color)
        fps_rect = fps_text.get_rect()
        fps_rect.bottomright = (self.screen.get_width() - 10, self.screen.get_height() - 10)
        self.screen.blit(fps_text, fps_rect)
    
    def _draw_effects(self):
        """Draw all particle effects"""
        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen)
        
        # Draw catch effects
        for effect in self.catch_effects:
            effect.draw(self.screen, self.font_medium)
    def _draw_game(self, inputs_state: torch.Tensor, game_state: torch.Tensor, last_action: int = None):
        """Draw the current game state with enhanced visuals"""
        # Update effects first
        self._update_effects()
        
        # Draw background
        self._draw_background()
        
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
        
        # Draw fruits with emoji and effects
        active_fruits = self._draw_fruits(fruits_data, scale_x, scale_y)
        
        # Draw player
        self._draw_player(sprite_pos, scale_x, scale_y)
        
        # Draw UI
        self._draw_ui(score, step_count, active_fruits, fruits_reached_bottom, last_action)
        
        # Draw particle effects
        self._draw_effects()
        
        # Check for fruit catching (simplified detection for effects)
        current_score = score
        if hasattr(self, 'previous_score') and current_score > self.previous_score:
            # Add catch effect at sprite position
            sprite_screen_x = int(sprite_pos * scale_x)
            sprite_screen_y = int((self.game_config.screen_height - 1) * scale_y)
            fruit_type = random.choice(self.fruit_types)
            self._add_catch_particles(sprite_screen_x, sprite_screen_y - 30, fruit_type)
        
        self.previous_score = current_score
        
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

class Particle:
    def __init__(self, x, y, color, velocity_x=0, velocity_y=0, life_time=60):
        self.x = x
        self.y = y
        self.color = color
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.life_time = life_time
        self.max_life = life_time
        self.size = random.randint(2, 5)
    
    def update(self):
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.velocity_y += 0.2  # gravity
        self.life_time -= 1
        return self.life_time > 0
    
    def draw(self, screen):
        alpha = int(255 * (self.life_time / self.max_life))
        color_with_alpha = (*self.color, alpha)
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)

class CatchEffect:
    def __init__(self, x, y, fruit_type):
        self.x = x
        self.y = y
        self.fruit_type = fruit_type
        self.life_time = 40
        self.max_life = 40
        self.scale = 1.0
    
    def update(self):
        self.life_time -= 1
        self.scale = 1.0 + (0.5 * (1 - self.life_time / self.max_life))
        return self.life_time > 0
    
    def draw(self, screen, font):
        alpha = int(255 * (self.life_time / self.max_life))
        text = f"🎉 +{self.fruit_type['name']}! 🎉"
        color = (*self.fruit_type['color'], min(alpha, 255))
        
        # Draw glowing text effect
        glow_surface = font.render(text, True, (255, 255, 255))
        text_surface = font.render(text, True, self.fruit_type['color'])
        
        screen.blit(text_surface, (self.x - text_surface.get_width()//2, self.y))