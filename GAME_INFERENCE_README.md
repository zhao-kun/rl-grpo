# Game Inference - Visual AI Demonstration

This module provides a visual demonstration of the trained AI playing the Fruits Catcher game using Pygame.

## Features

- **Real-time Visualization**: Watch the AI make decisions in real-time
- **Detailed Debug Information**: See scores, steps, active fruits, and AI actions
- **Configurable Display**: Adjustable scaling factors for better visibility
- **Interactive Controls**: ESC to quit, visual feedback

## Files

- `game_inference.py`: Main GameInference class implementation
- `run_game_demo.py`: Simple demo script to run the game
- `test_inference.py`: Test script with detailed configuration

## How It Works

### Game State Representation
The game state consists of:
- **Sprite Position**: X-coordinate of the player-controlled sprite
- **Fruits Data**: For each fruit (up to max_fruits_on_screen):
  - X position (0 to screen_width-1)
  - Y position (0 to screen_height-1) 
  - Active status (1.0 = active, 0.0 = inactive)

### Game Loop
1. **Initialize**: Create initial game state with sprite and first fruit
2. **Update**: Every `refresh_timer` milliseconds:
   - AI brain decides action (LEFT=0, STAY=1, RIGHT=2)
   - GameEngine updates positions and spawns new fruits
   - Collision detection and scoring
3. **Render**: Draw game state with visual scaling
4. **Repeat**: Until game over condition

### Visual Elements
- **Red Circles**: Active falling fruits
- **Green Rectangle**: AI-controlled sprite (player)
- **Grid Lines**: Game boundaries (optional)
- **UI Text**: Score, steps, debug information

### Scaling System
All visual elements are scaled according to:
- `view_width_multiplier`: Horizontal scaling factor
- `view_height_multiplier`: Vertical scaling factor

This allows the logical game grid (e.g., 20x11) to be displayed at any resolution.

## Usage

### Basic Usage
```python
from grpo_fruits_catcher import GameConfig, TrainerConfig
from game_inference import GameInference

# Create configuration
config = TrainerConfig(game_config=GameConfig())

# Run inference
game = GameInference("model.pth", config, device='cuda')
game.run()
```

### Configuration Options

#### Game Configuration
- `screen_width/height`: Logical game grid dimensions
- `sprite_width/height`: Player sprite size
- `max_fruits_on_screen`: Maximum simultaneous fruits
- `min_fruits_on_screen`: Minimum fruits to maintain
- `min_interval_step_fruits`: Minimum spacing between fruits
- `view_width/height_multiplier`: Visual scaling factors
- `refresh_timer`: Update interval in milliseconds
- `ended_game_score`: Game over threshold

#### Recommended Settings for Demonstration
```python
game_config = GameConfig(
    screen_width=20,
    screen_height=11,
    view_width_multiplier=25.0,   # 25x scaling
    view_height_multiplier=25.0,  # 25x scaling
    refresh_timer=150,            # 150ms updates
    ended_game_score=-30          # End at -30 points
)
```

## Running the Demo

### Option 1: Simple Demo
```bash
python3 run_game_demo.py
```

### Option 2: Custom Configuration
```bash
python3 test_inference.py
```

### Option 3: Programmatic Usage
```python
import torch
from grpo_fruits_catcher import TrainerConfig, GameConfig
from game_inference import GameInference

# Setup
config = TrainerConfig(
    game_config=GameConfig(
        view_width_multiplier=30.0,
        view_height_multiplier=30.0,
        refresh_timer=200
    )
)

# Run
device = 'cuda' if torch.cuda.is_available() else 'cpu'
game = GameInference("grpo_fruits_catcher-002000.pth", config, device)
game.run()
```

## Controls

- **ESC**: Quit the game
- **Close Window**: Quit the game
- The AI automatically controls the sprite movement

## Debug Information Displayed

- **Score**: Current game score (catches - misses)
- **Steps**: Number of game updates executed
- **Active Fruits**: Number of fruits currently falling
- **Fruits Reached Bottom**: Total fruits that reached the bottom
- **Last Action**: AI's most recent decision (LEFT/STAY/RIGHT)
- **Sprite Position**: Current X-coordinate of the player sprite

## Performance Notes

- Uses GPU acceleration when available (`cuda`)
- Optimized for real-time performance
- 60 FPS display with configurable game update intervals
- Efficient tensor operations for game state management

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure `grpo_fruits_catcher-XXXXXX.pth` exists
2. **Display Issues**: Try different scaling multipliers
3. **Performance**: Use CPU device if GPU causes issues
4. **Audio Warnings**: ALSA warnings are harmless (no audio needed)

### System Requirements
- Python 3.8+
- PyTorch
- Pygame
- CUDA-capable GPU (optional, recommended)
