import torch
import pygame
import math
import random
from typing import Tuple, List, Dict
from grpo_fruits_catcher import GameConfig, Trainer, TrainerConfig, GameBrain, GameEngine


class GameInference:
    device: str = 'cpu'
    game_config: GameConfig = None
    trainer_config: TrainerConfig = None
    gb: GameBrain = None
    engine: GameEngine = None
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize GameInference with automatic config loading from saved model.
        
        Args:
            model_path: Path to the saved model file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        assert model_path is not None, "Model path must be provided"
        self.device = device
        self.model_path = model_path
        
        # Load model and configs from the saved checkpoint
        self.gb, self.game_config, self.trainer_config = GameBrain.from_pretrained(model_path, device)
        self.engine = GameEngine(self.trainer_config, self.gb)

        if self.game_config.view_height_multiplier < 50.0:
            self.game_config.view_height_multiplier = 50.0
        if self.game_config.view_width_multiplier < 50.0:
            self.game_config.view_width_multiplier = 50.0
        if self.game_config.refresh_timer < 100:
            self.game_config.refresh_timer = 100
        
        # Initialize score tracking for effects
        self.previous_score = 0
        
        # Game ending state tracking
        self.game_ended = False
        self.game_result = None  # 'win', 'lose', or None
        self.game_end_time = None  # Track when game ended for auto-exit
    
    @classmethod
    def from_pretrained(cls, model_path: str, device: str = 'cpu') -> 'GameInference':
        """
        Create GameInference instance from a pretrained model.
        
        Args:
            model_path: Path to the saved model file
            device: Device to run inference on
            
        Returns:
            GameInference instance with loaded model and configs
        """
        return cls(model_path, device)
    
    
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
        # Try to use a color emoji font for fruit emojis
        import os
        emoji_font_path = None
        # Common locations for Noto Color Emoji on Linux
        noto_paths = [
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
            "/usr/share/fonts/NotoColorEmoji.ttf",
            "/usr/local/share/fonts/NotoColorEmoji.ttf"
        ]
        for path in noto_paths:
            if os.path.exists(path):
                emoji_font_path = path
                break
        if emoji_font_path:
            self.font_emoji = pygame.font.Font(emoji_font_path, 24)
            self.font_emoji_ui = pygame.font.Font(emoji_font_path, 8)  # Even smaller font for UI emojis
            print(f"[Debug] Using emoji font: {emoji_font_path}, UI size: 8px")
        else:
            # Fallback to default font (may not support emoji)
            self.font_emoji = pygame.font.Font(None, 24)
            self.font_emoji_ui = pygame.font.Font(None, 8)  # Even smaller font for UI emojis
            print("[Warning] Noto Color Emoji font not found. Fruit emojis may not display correctly.")
            print("[Debug] Using default font for UI emojis, size: 8px")
        
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
                
                # Draw fruit glow effect (reduced size)
                glow_radius = int(min(scale_x, scale_y) // 2.5)
                for r in range(glow_radius, 0, -2):
                    alpha = int(30 * (1 - r / glow_radius))
                    glow_color = (*fruit_type['color'], alpha)
                    pygame.draw.circle(self.screen, fruit_type['color'], 
                                     (center_x, center_y), r)
                
                # Draw fruit emoji (scaled down)
                emoji_surface = self.font_emoji.render(fruit_type['emoji'], True, self.WHITE)
                # Scale down the emoji surface by 1.5 factor
                original_size = emoji_surface.get_size()
                new_size = (int(original_size[0] / 1.5), int(original_size[1] / 1.5))
                emoji_surface = pygame.transform.scale(emoji_surface, new_size)
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
        # Fix emoji display for score - use smaller emoji and scale it down further
        trophy_emoji_raw = self.font_emoji_ui.render("🏆", True, score_color)
        # Scale it down even more
        trophy_size = trophy_emoji_raw.get_size()
        new_trophy_size = (int(trophy_size[0] * 0.3), int(trophy_size[1] * 0.3))
        trophy_emoji = pygame.transform.scale(trophy_emoji_raw, new_trophy_size)
        score_only_text = pulse_font.render(f"Score: {score:.1f}", True, score_color)
        
        # Align trophy emoji vertically with score text
        text_center_y = 20 + score_only_text.get_height() // 2
        emoji_y = text_center_y - trophy_emoji.get_height() // 2
        self.screen.blit(trophy_emoji, (20, emoji_y))
        self.screen.blit(score_only_text, (20 + trophy_emoji.get_width() + 8, 20))
        
        # Game stats with icons
        stats_y = 80
        line_height = 35  # Fixed line height for consistent spacing
        
        # Fix emoji display for stats - use smaller emojis, scale them down, and better positioning
        timer_emoji_raw = self.font_emoji_ui.render("⏱️", True, self.WHITE)
        timer_size = timer_emoji_raw.get_size()
        timer_emoji = pygame.transform.scale(timer_emoji_raw, (int(timer_size[0] * 0.3), int(timer_size[1] * 0.3)))
        step_only_text = self.font_medium.render(f"Steps: {step_count}", True, self.WHITE)
        # Align emoji vertically with text center
        text_center_y = stats_y + step_only_text.get_height() // 2
        emoji_y = text_center_y - timer_emoji.get_height() // 2
        self.screen.blit(timer_emoji, (20, emoji_y))
        self.screen.blit(step_only_text, (20 + timer_emoji.get_width() + 8, stats_y))
        
        apple_emoji_raw = self.font_emoji_ui.render("🍎", True, self.WHITE)
        apple_size = apple_emoji_raw.get_size()
        apple_emoji = pygame.transform.scale(apple_emoji_raw, (int(apple_size[0] * 0.3), int(apple_size[1] * 0.3)))
        fruits_only_text = self.font_medium.render(f"Active: {active_fruits}", True, self.WHITE)
        # Align emoji vertically with text center
        text_center_y = stats_y + line_height + fruits_only_text.get_height() // 2
        emoji_y = text_center_y - apple_emoji.get_height() // 2
        self.screen.blit(apple_emoji, (20, emoji_y))
        self.screen.blit(fruits_only_text, (20 + apple_emoji.get_width() + 8, stats_y + line_height))
        
        missed_color = self.RED if fruits_reached_bottom > 0 else self.WHITE
        boom_emoji_raw = self.font_emoji_ui.render("💥", True, missed_color)
        boom_size = boom_emoji_raw.get_size()
        boom_emoji = pygame.transform.scale(boom_emoji_raw, (int(boom_size[0] * 0.3), int(boom_size[1] * 0.3)))
        bottom_only_text = self.font_medium.render(f"Missed: {fruits_reached_bottom}", True, missed_color)
        # Align emoji vertically with text center
        text_center_y = stats_y + line_height * 2 + bottom_only_text.get_height() // 2
        emoji_y = text_center_y - boom_emoji.get_height() // 2
        self.screen.blit(boom_emoji, (20, emoji_y))
        self.screen.blit(bottom_only_text, (20 + boom_emoji.get_width() + 8, stats_y + line_height * 2))
        
        # Action indicator with animation - FIXED POSITION
        if last_action is not None:
            action_names = ["⬅️ LEFT", "⏸️ STAY", "➡️ RIGHT"]
            action_emojis = ["⬅️", "⏸️", "➡️"]
            action_texts = ["LEFT", "STAY", "RIGHT"]
            action_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
            
            # Render gamepad emoji and action separately with smaller size and scaling
            gamepad_emoji_raw = self.font_emoji_ui.render("🎮", True, action_colors[last_action])
            gamepad_size = gamepad_emoji_raw.get_size()
            gamepad_emoji = pygame.transform.scale(gamepad_emoji_raw, (int(gamepad_size[0] * 0.3), int(gamepad_size[1] * 0.3)))
            
            action_emoji_raw = self.font_emoji_ui.render(action_emojis[last_action], True, action_colors[last_action])
            action_emoji_size = action_emoji_raw.get_size()
            action_emoji = pygame.transform.scale(action_emoji_raw, (int(action_emoji_size[0] * 0.3), int(action_emoji_size[1] * 0.3)))
            
            action_text_part = self.font_medium.render(action_texts[last_action], True, action_colors[last_action])
            
            # FIXED POSITION - no more moving based on text length
            fixed_width = 120  # Fixed width for the action indicator
            start_x = self.screen.get_width() - 100 - fixed_width
            
            # Add action background with fixed size
            bg_height = max(gamepad_emoji.get_height(), action_text_part.get_height()) + 8
            bg_rect = pygame.Rect(start_x - 5, 18, fixed_width + 70, bg_height)
            pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect)
            pygame.draw.rect(self.screen, action_colors[last_action], bg_rect, 2)
            
            # Calculate vertical center for alignment
            bg_center_y = 18 + bg_height // 2
            
            # Blit gamepad emoji at fixed position
            gamepad_y = bg_center_y - gamepad_emoji.get_height() // 2
            self.screen.blit(gamepad_emoji, (start_x + 5, gamepad_y))
            
            # Blit action emoji at fixed position with more spacing
            action_emoji_y = bg_center_y - action_emoji.get_height() // 2
            self.screen.blit(action_emoji, (start_x + 5 + 45, action_emoji_y))  # More spacing
            
            # Blit text at fixed position with more spacing
            text_y = bg_center_y - action_text_part.get_height() // 2
            self.screen.blit(action_text_part, (start_x + 5 + 45 + 45, text_y))  # More spacing for text
        
        # AI status indicator
        robot_emoji_raw = self.font_emoji_ui.render("🤖", True, self.CYAN)
        robot_size = robot_emoji_raw.get_size()
        robot_emoji = pygame.transform.scale(robot_emoji_raw, (int(robot_size[0] * 0.3), int(robot_size[1] * 0.3)))
        ai_text_part = self.font_small.render("AI PLAYING - Press ESC to quit", True, self.CYAN)
        
        # Position at bottom with proper vertical alignment
        bottom_y = self.screen.get_height() - 15
        text_y = bottom_y - ai_text_part.get_height()
        emoji_y = text_y + ai_text_part.get_height() // 2 - robot_emoji.get_height() // 2
        
        self.screen.blit(robot_emoji, (20, emoji_y))
        self.screen.blit(ai_text_part, (20 + robot_emoji.get_width() + 5, text_y))
        
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
        print(f"Game will end when score reaches {self.game_config.win_ended_game_score} (WIN) or {self.game_config.fail_ended_game_score} (LOSE)")
        
        while running:
            current_time = pygame.time.get_ticks()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif self.game_ended:
                        # Any key press after game ends will quit
                        print("Game ended by user input")
                        running = False
            
            # Update game state at specified intervals (only if game hasn't ended)
            if current_time - last_update_time >= self.game_config.refresh_timer and not self.game_ended:
                # Update game using AI brain
                new_inputs_state, actions, new_game_state = self.engine.update(inputs_state, game_state)
                inputs_state = new_inputs_state
                game_state = new_game_state
                last_action = actions[0, 0].item()  # Extract action for single game
                last_update_time = current_time
                
                # Get current score
                current_score = game_state[0, 0, 0].item()
                
                # Check win/lose conditions
                if current_score >= self.game_config.win_ended_game_score:
                    print(f"🏆 AI WINS! Final Score: {current_score:.1f}")
                    self.game_ended = True
                    self.game_result = 'win'
                    self.game_end_time = current_time
                elif current_score <= self.game_config.fail_ended_game_score:
                    print(f"💥 AI LOSES! Final Score: {current_score:.1f}")
                    self.game_ended = True
                    self.game_result = 'lose'
                    self.game_end_time = current_time
                
                # Optional: print periodic updates (only when game is still running)
                step_count = int(game_state[0, 0, 1].item())
                if step_count % 50 == 0:  # Print every 50 steps
                    print(f"Step {step_count}: Score = {current_score:.1f}")
            
            # Draw the game
            self._draw_game(inputs_state, game_state, last_action)
            
            # Draw ending screen if game is over
            if self.game_ended:
                current_score = game_state[0, 0, 0].item()
                step_count = int(game_state[0, 0, 1].item())
                
                if self.game_result == 'win':
                    self._draw_win_screen(current_score, step_count)
                elif self.game_result == 'lose':
                    self._draw_lose_screen(current_score, step_count)
                
                # Auto-exit after 10 seconds of showing end screen
                if self.game_end_time and (current_time - self.game_end_time) > 10000:  # 10 seconds
                    print("Auto-exiting after 10 seconds...")
                    running = False
            
            # Control frame rate
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
        print("Game finished!")

    def _draw_win_screen(self, score, steps):
        """Draw the victory screen when AI wins"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()))
        overlay.set_alpha(200)
        overlay.fill((0, 50, 0))  # Dark green overlay
        self.screen.blit(overlay, (0, 0))
        
        # Victory title
        title_text = "🏆 AI VICTORY! 🏆"
        title_surface = self.font_large.render(title_text, True, (255, 215, 0))  # Gold color
        title_rect = title_surface.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()//3))
        
        # Add glow effect to title
        for offset in [(2, 2), (-2, -2), (2, -2), (-2, 2)]:
            glow_surface = self.font_large.render(title_text, True, (255, 255, 255))
            glow_rect = glow_surface.get_rect(center=(title_rect.centerx + offset[0], title_rect.centery + offset[1]))
            self.screen.blit(glow_surface, glow_rect)
        
        self.screen.blit(title_surface, title_rect)
        
        # Victory stats
        stats_y = self.screen.get_height()//2
        stats = [
            f"🎯 Final Score: {score:.1f}",
            f"⏱️ Steps Taken: {steps}",
            f"🤖 AI Performance: EXCELLENT!",
            f"🌟 Target Reached: {self.game_config.win_ended_game_score}"
        ]
        
        for i, stat in enumerate(stats):
            stat_surface = self.font_medium.render(stat, True, (200, 255, 200))
            stat_rect = stat_surface.get_rect(center=(self.screen.get_width()//2, stats_y + i*40))
            self.screen.blit(stat_surface, stat_rect)
        
        # Victory message
        message = "The AI has successfully mastered the Fruits Catcher game!"
        message_surface = self.font_small.render(message, True, (255, 255, 255))
        message_rect = message_surface.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()*3//4))
        self.screen.blit(message_surface, message_rect)
        
        # Exit instruction
        exit_text = "Press any key to exit (auto-exit in 10 seconds)"
        exit_surface = self.font_small.render(exit_text, True, (200, 200, 200))
        exit_rect = exit_surface.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()*7//8))
        self.screen.blit(exit_surface, exit_rect)
    
    def _draw_lose_screen(self, score, steps):
        """Draw the game over screen when AI loses"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()))
        overlay.set_alpha(200)
        overlay.fill((50, 0, 0))  # Dark red overlay
        self.screen.blit(overlay, (0, 0))
        
        # Game over title
        title_text = "💥 GAME OVER 💥"
        title_surface = self.font_large.render(title_text, True, (255, 100, 100))  # Red color
        title_rect = title_surface.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()//3))
        
        # Add glow effect to title
        for offset in [(2, 2), (-2, -2), (2, -2), (-2, 2)]:
            glow_surface = self.font_large.render(title_text, True, (255, 255, 255))
            glow_rect = glow_surface.get_rect(center=(title_rect.centerx + offset[0], title_rect.centery + offset[1]))
            self.screen.blit(glow_surface, glow_rect)
        
        self.screen.blit(title_surface, title_rect)
        
        # Failure stats
        stats_y = self.screen.get_height()//2
        stats = [
            f"💔 Final Score: {score:.1f}",
            f"⏱️ Steps Taken: {steps}",
            f"🤖 AI Performance: NEEDS IMPROVEMENT",
            f"🎯 Failure Threshold: {self.game_config.fail_ended_game_score}"
        ]
        
        for i, stat in enumerate(stats):
            stat_surface = self.font_medium.render(stat, True, (255, 200, 200))
            stat_rect = stat_surface.get_rect(center=(self.screen.get_width()//2, stats_y + i*40))
            self.screen.blit(stat_surface, stat_rect)
        
        # Failure message
        message = "The AI needs more training to master this game!"
        message_surface = self.font_small.render(message, True, (255, 255, 255))
        message_rect = message_surface.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()*3//4))
        self.screen.blit(message_surface, message_rect)
        
        # Exit instruction
        exit_text = "Press any key to exit (auto-exit in 10 seconds)"
        exit_surface = self.font_small.render(exit_text, True, (200, 200, 200))
        exit_rect = exit_surface.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()*7//8))
        self.screen.blit(exit_surface, exit_rect)

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