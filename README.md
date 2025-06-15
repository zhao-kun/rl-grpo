# 🍎 RL-GRPO Fruits Catcher

A reinforcement learning project using Group Relative Policy Optimization (GRPO) to train an AI agent to play a fruits catching game. The AI learns to control a sprite that catches falling fruits while avoiding negative scores.

This implementation demonstrates the principles of GRPO, the algorithm proposed by DeepSeek for post-training the DeepSeek-R1 model, applied to a simple game environment.

<div align="center">
  <img src="assets/demo_game.gif" alt="GRPO Fruits Catcher Demo" width="600" />
  <p><em>🤖 AI agent trained with GRPO playing the fruits catching game</em></p>
</div>

> **Note:** This is an educational project (~50% AI-assisted) by a non-ML expert. Please see the [Important Disclaimer](#️-important-disclaimer) section for details.

## 🙏 Acknowledgments

This project is inspired by **"How does DeepSeek learn? GRPO explained with Triangle Creatures"** by **Dr. Mihai Nica** on YouTube: [https://www.youtube.com/watch?v=wXEvvg4YJ9I](https://www.youtube.com/watch?v=wXEvvg4YJ9I)

**Why a different implementation?**
- **Original**: JAX-based with Triangle Creatures (complex movement mechanics)
- **This version**: PyTorch-based with Fruits Catching (simpler, more accessible logic)

The original Triangle Creatures implementation demonstrated GRPO beautifully but had complex movement mechanics. This fruits catching version simplifies the game logic while preserving the core GRPO learning principles, making it more accessible for educational purposes and PyTorch users.

**Special thanks to Dr. Mihai Nica** for the excellent educational content and original GRPO implementation that inspired this project!

## ⚠️ Important Disclaimer

**AI-Assisted Development:** Approximately 50% of this codebase was written with assistance from GitHub Copilot Agent (Claude Sonnet 4 Preview). 

**Author's Note:** I am not an expert in the ML domain, so I apologize if the code contains incorrect content or suboptimal implementations. This project is primarily intended for educational purposes and learning GRPO concepts.

**Recommendations:**
- Use this as a learning resource rather than production code
- Verify implementations against academic sources when in doubt
- Contributions from ML experts are especially welcome to improve accuracy
- Always cross-reference with the original DeepSeek papers and Dr. Mihai Nica's work

## 📚 Table of Contents

- [🙏 Acknowledgments](#-acknowledgments)
- [⚠️ Important Disclaimer](#️-important-disclaimer)
- [🌟 Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)  
- [🎮 Game Mechanics](#-game-mechanics)
- [🧠 Training Guide](#-training-guide)
- [🎮 Game Demo Guide](#-game-demo-guide)
- [🏗️ Architecture](#️-architecture)
- [📁 Project Structure](#-project-structure)
- [🔬 Research & Experimentation](#-research--experimentation)
- [📚 About GRPO](#-about-grpo)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📚 References](#-references)

## 🌟 Features

- **🤖 AI-Controlled Gameplay**: Watch the trained AI play the fruits catching game
- **🧠 GRPO Training**: Group Relative Policy Optimization algorithm with policy optimization
- **🎮 Customizable Game**: Configurable screen size, fruit spawn rates, scoring thresholds
- **⚡ PyTorch Compilation**: Optional torch.compile for faster training
- **🛑 Early Stopping**: Intelligent training termination with patience control
- **📊 Comprehensive Logging**: Detailed training progress and configuration display
- **🎯 Configuration Overrides**: Runtime game parameter adjustments without retraining

## 🚀 Quick Start

### 1. Installation
```bash
git clone <repository-url>
cd rl-grpo
uv sync
source .venv/bin/activate  # Activate virtual environment
```

### 2. Training a Model
```bash
# Default training (recommended)
python main.py

# Quick test training
python main.py --total-epochs 10 --batch-size 4
```

### 3. Running the Game Demo
```bash
# Run with latest trained model
python run_game_demo.py

# Run with specific model
python run_game_demo.py --model grpo_fruits_catcher-002000.pth
```

## 📦 Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast and reliable dependency management. uv is a modern Python package manager that's much faster than pip and provides better dependency resolution.

### Prerequisites
- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Install uv (if not already installed)
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

### Install with uv (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd rl-grpo

# Install dependencies using uv
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows
```

### Development Installation
For development with testing dependencies:
```bash
# Install with development dependencies
uv sync --group dev

# Run tests
uv run pytest
```

### Alternative Installation (pip)
If you prefer using pip:
```bash
# Clone the repository
git clone <repository-url>
cd rl-grpo

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install torch pygame numpy tqdm matplotlib pytest
```

### Verify Installation
```bash
# Test the installation
python main.py --help
python run_game_demo.py --help
```

### Troubleshooting
If you encounter issues:

1. **Python Version**: Ensure you're using Python 3.10 or higher
   ```bash
   python --version
   ```

2. **Virtual Environment**: Make sure the virtual environment is activated
   ```bash
   source .venv/bin/activate  # Linux/macOS
   ```

3. **Dependencies**: If using pip instead of uv, install exact versions:
   ```bash
   pip install torch pygame==2.6.1 numpy==1.26.4 tqdm matplotlib pytest
   ```

4. **GPU Support**: For CUDA support, install PyTorch with CUDA:
   ```bash
   # For CUDA 11.8
   uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 🎮 Game Mechanics

- **🍎 Fruits**: Randomly spawn and fall down the screen
- **🤖 AI Sprite**: Green rectangle that moves left/right to catch fruits
- **📈 Scoring**: +1 for catching fruits, -1 for missing them
- **🏆 Win Condition**: Reach +30 score
- **💥 Lose Condition**: Drop to -30 score
- **⏱️ Time Limit**: Configurable maximum steps per episode

---

# 🧠 Training Guide

## 🚀 Quick Training Examples

### Default Training
```bash
python main.py
```

### Custom Training Examples

#### 🎯 Quick Test Training (1 epoch)
```bash
python main.py --total-epochs 1 --batch-size 4
```

#### ⚡ Fast Training with Compilation
```bash
python main.py --compile --total-epochs 1000 --lr-rate 2e-4
```

#### 🎮 Custom Game Configuration
```bash
python main.py \
  --screen-width 25 \
  --screen-height 20 \
  --max-fruits 5 \
  --win-score 50 \
  --fail-score -50
```

#### 🧠 Large Model Training
```bash
python main.py \
  --hidden-size 4096 \
  --batch-size 64 \
  --total-epochs 3000 \
  --lr-rate 5e-5 \
  --max-steps 150 \
  --patience 800
```

#### 🛑 Early Stopping Control
```bash
# Quick testing with early stopping after 50 epochs
python main.py --total-epochs 1000 --patience 50

# Conservative training with longer patience
python main.py --total-epochs 5000 --patience 500

# Aggressive early stopping for quick experiments
python main.py --total-epochs 2000 --patience 100
```

#### 💻 CPU Training
```bash
python main.py --device cpu --batch-size 8 --total-epochs 500
```

#### 📂 Custom Model Name
```bash
python main.py --model-name my_custom_model --total-epochs 1500
```

## 📋 All Training Arguments

### 🎮 Game Configuration
- `--screen-width` - Game screen width (default: 20)
- `--screen-height` - Game screen height (default: 15) 
- `--sprite-width` - AI sprite width (default: 3)
- `--sprite-height` - AI sprite height (default: 1)
- `--max-fruits` - Maximum fruits on screen (default: 3)
- `--min-fruits` - Minimum fruits on screen (default: 1)
- `--min-interval-steps` - Minimum steps between fruit spawns (default: 4)
- `--view-height-multiplier` - View height scaling factor (default: 50.0)
- `--view-width-multiplier` - View width scaling factor (default: 50.0)
- `--refresh-timer` - Game refresh timer in ms (default: 150)
- `--fail-score` - Score threshold for game failure (default: -30)
- `--win-score` - Score threshold for game victory (default: 30)

### 🧠 Training Configuration  
- `--hidden-size` - Neural network hidden layer size (default: 2048)
- `--batch-size` - Training batch size (default: 32)
- `--total-epochs` - Total training epochs (default: 2000)
- `--max-steps` - Maximum steps per episode (default: 100)
- `--lr-rate` - Learning rate (default: 1e-4)
- `--patience` - Early stopping patience in epochs (default: 500)
- `--compile` - Enable torch.compile for faster training
- `--no-compile` - Disable torch.compile (default)

### 💾 Output Configuration
- `--model-name` - Model save name (default: grpo_fruits_catcher)
- `--device` - Training device: auto, cpu, cuda, cuda:0, cuda:1 (default: auto)

## 💡 Training Tips

### 🎯 For Quick Testing
- Use `--total-epochs 1-10` for quick validation
- Use `--batch-size 2-4` for faster iterations

### 🏆 For Best Performance  
- Use `--compile` for faster training (PyTorch 2.0+)
- Use `--hidden-size 1024` or higher for complex games
- Use `--batch-size 32` or higher if you have enough GPU memory

### 🎮 For Custom Games
- Increase `--win-score` and decrease `--fail-score` for longer episodes
- Increase `--max-fruits` for more challenging gameplay
- Adjust `--max-steps` based on your game difficulty

### 🛑 Early Stopping Guide

The `--patience` parameter controls when training stops if no improvement is seen:

- **`--patience 100`**: Stops if no improvement for 100 epochs (quick experiments)
- **`--patience 300`**: Good for medium-length training sessions
- **`--patience 500`**: Default value, good balance between efficiency and thoroughness
- **`--patience 1000`**: Very patient, suitable for complex models/games

**When to adjust patience:**
- **Short patience (50-100)**: Testing, debugging, quick experiments
- **Medium patience (200-400)**: Normal training, most use cases
- **Long patience (500+)**: Complex games, large models, research

## 📊 Example Training Configurations

### Beginner (Fast Training)
```bash
python main.py --total-epochs 500 --batch-size 8 --hidden-size 512 --patience 100
```

### Intermediate (Balanced)
```bash  
python main.py --total-epochs 1500 --batch-size 16 --hidden-size 1024 --compile --patience 300
```

### Advanced (High Performance)
```bash
python main.py --total-epochs 3000 --batch-size 32 --hidden-size 2048 --compile --lr-rate 5e-5 --patience 500
```

### Research (Long Training)
```bash
python main.py --total-epochs 5000 --batch-size 64 --hidden-size 4096 --max-steps 200 --compile --patience 1000
```

---

# 🎮 Game Demo Guide

## 🚀 Quick Start

### Default Usage
```bash
python run_game_demo.py
```
Automatically finds the latest trained model and runs the game with full instructions.

> 🎬 **See the demo animation at the top of this README** to get a preview of what the trained AI looks like in action!

## 📋 All Demo Arguments

### 🤖 Model Configuration

#### `--model, -m`
Specify exact model file path
```bash
python run_game_demo.py --model grpo_fruits_catcher-002000.pth
python run_game_demo.py -m my_custom_model-001500.pth
```

#### `--model-name`
Model name prefix to search for (default: `grpo_fruits_catcher`)
```bash
python run_game_demo.py --model-name my_custom_model
python run_game_demo.py --model-name experimental_v2
```

#### `--device`
Computation device (choices: `auto`, `cpu`, `cuda`, default: `auto`)
```bash
python run_game_demo.py --device cpu      # Force CPU usage
python run_game_demo.py --device cuda     # Force CUDA usage  
python run_game_demo.py --device auto     # Auto-detect (default)
```

### 🎮 Game Configuration Overrides

#### `--min-interval-steps`
Override minimum steps between fruit spawns (overrides model's saved configuration)
```bash
python run_game_demo.py --min-interval-steps 2   # Faster fruit spawning
python run_game_demo.py --min-interval-steps 8   # Slower fruit spawning
python run_game_demo.py --min-interval-steps 1   # Maximum fruit spawn rate
```

### 📊 Display Configuration

#### `--verbose, -v`
Show detailed model and game configuration
```bash
python run_game_demo.py --verbose
python run_game_demo.py -v
```

#### `--config-only`
Only display configuration without running the game
```bash
python run_game_demo.py --config-only          # Brief config
python run_game_demo.py --verbose --config-only # Detailed config
```

#### `--quiet, -q`
Minimal output (no instructions or verbose info)
```bash
python run_game_demo.py --quiet
python run_game_demo.py -q
```

## 🎯 Demo Usage Examples

### 🔍 **Inspect Model Configuration**
```bash
# Quick config check
python run_game_demo.py --config-only

# Detailed configuration analysis
python run_game_demo.py --verbose --config-only

# Check specific model
python run_game_demo.py --model my_model-001000.pth --verbose --config-only
```

### 🎮 **Run Specific Models**
```bash
# Run latest model with full output
python run_game_demo.py

# Run specific model quietly
python run_game_demo.py --model grpo_fruits_catcher-002000.pth --quiet

# Run with verbose info first, then game
python run_game_demo.py --verbose
```

### 🔧 **Development & Testing**
```bash
# Test on CPU only
python run_game_demo.py --device cpu --quiet

# Check different model series
python run_game_demo.py --model-name experimental --verbose --config-only

# Quick test run with minimal output
python run_game_demo.py --model grpo_fruits_catcher-000005.pth -q

# Test with faster fruit spawning
python run_game_demo.py --min-interval-steps 2 --verbose

# Test with much slower fruit spawning
python run_game_demo.py --min-interval-steps 10
```

### 📊 **Model Comparison**
```bash
# Compare different models
python run_game_demo.py --model model_v1-002000.pth --verbose --config-only
python run_game_demo.py --model model_v2-002000.pth --verbose --config-only

# Test model series performance
python run_game_demo.py --model-name model_v1 --quiet
python run_game_demo.py --model-name model_v2 --quiet
```

### 🎮 **Gameplay Tuning**
```bash
# Make game easier (slower fruit spawning)
python run_game_demo.py --min-interval-steps 8

# Make game harder (faster fruit spawning)  
python run_game_demo.py --min-interval-steps 2

# Extreme challenge mode (maximum spawn rate)
python run_game_demo.py --min-interval-steps 1

# Compare AI performance with different difficulty levels
python run_game_demo.py --min-interval-steps 2 --verbose --config-only
python run_game_demo.py --min-interval-steps 8 --verbose --config-only
```

## 📝 Verbose Configuration Display

When using `--verbose`, you'll see:

### 🎮 Game Configuration
- 📏 Screen dimensions and sprite size
- 🍎 Fruit spawn parameters  
- 📐 View scaling factors
- 🔄 Refresh rate settings
- 🎯 Win/lose score thresholds
- ⚙️ Configuration overrides (marked with "OVERRIDDEN" when present)

### 🧠 Training Configuration  
- 🔄 Total training epochs
- 📦 Batch size used
- 🧠 Neural network architecture
- 📈 Learning rate and training settings
- ⚡ Compilation status
- 🏗️ Model parameter counts
- 📥📤 Input/output dimensions

## 🎛️ Model Search Behavior

### **When `--model` is specified:**
- Uses exact file path
- Shows error if not found

### **When `--model-name` is used:**
- Searches for files starting with the prefix
- Automatically selects highest epoch number
- Falls back to default candidates if none found

### **Default search order:**
1. Search for `{model_name}-*.pth` files
2. Sort by epoch number (highest first)
3. If none found, try:
   - `{model_name}-003000.pth`
   - `{model_name}-002000.pth`
   - `{model_name}-001000.pth`
   - `{model_name}-000001.pth`

## 💡 Pro Tips

### 🎯 **For Quick Testing**
```bash
# Minimal run
python run_game_demo.py -q

# Config check only
python run_game_demo.py --config-only
```

### 🔍 **For Analysis**
```bash
# Full model analysis
python run_game_demo.py -v --config-only

# Compare configurations
python run_game_demo.py --model old_model.pth -v --config-only
python run_game_demo.py --model new_model.pth -v --config-only
```

### 🎮 **For Demos**
```bash
# Clean demo run
python run_game_demo.py

# Demo with background info
python run_game_demo.py --verbose
```

### 🛠️ **For Development**
```bash
# Test latest changes
python run_game_demo.py --model-name latest_experiment -v

# CPU testing
python run_game_demo.py --device cpu -q

# Test gameplay balance
python run_game_demo.py --min-interval-steps 1 --verbose  # Hard mode
python run_game_demo.py --min-interval-steps 10 --verbose # Easy mode
```

## 🎊 Output Modes

| Mode | Instructions | Configuration | Game Run |
|------|-------------|---------------|----------|
| **Default** | ✅ Full | ❌ Brief | ✅ Yes |
| **`--verbose`** | ✅ Full | ✅ Detailed | ✅ Yes |
| **`--quiet`** | ❌ None | ❌ None | ✅ Yes |
| **`--config-only`** | ❌ None | ✅ Brief | ❌ No |
| **`--verbose --config-only`** | ❌ None | ✅ Detailed | ❌ No |

Choose the mode that fits your needs for testing, analysis, or demonstration!

---

## 🏗️ Architecture

### 🧠 Neural Network
- **Input**: Game state (fruit positions, sprite position, score, etc.)
- **Hidden Layer**: Configurable size (default 2048 neurons)
- **Output**: Action probabilities (left, stay, right)
- **Activation**: GELU with layer normalization
- **Regularization**: Dropout and L2 regularization

### 🎯 GRPO Algorithm
- **Policy Optimization**: Group Relative Policy Optimization as proposed by DeepSeek
- **Reward Shaping**: Balanced positive/negative rewards with clipping
- **Entropy Bonus**: Encourages exploration
- **Return Normalization**: Stabilizes training
- **Gradient Clipping**: Prevents training instability

### 🛑 Training Features
- **Early Stopping**: Configurable patience with best model restoration
- **Learning Rate Scheduling**: Conservative decay for stability
- **Compilation Support**: Optional torch.compile for speed
- **Comprehensive Logging**: Progress tracking and verbose output

## 📁 Project Structure

```
rl-grpo/
├── main.py                    # Training script
├── run_game_demo.py          # Game demo script  
├── grpo_fruits_catcher.py    # Core GRPO implementation
├── game_inference.py         # Game inference engine
├── pyproject.toml            # Project configuration and dependencies
├── uv.lock                   # Dependency lock file (uv)
├── README.md                 # This file
├── assets/                   # Demo animations and media
│   └── demo_game.gif         # Game demo animation
├── REWARD_ALGORITHM.md       # Reward system documentation
├── GAME_INFERENCE_README.md  # Game inference documentation
├── pytest.ini               # Test configuration
├── tests/                    # Test files
│   ├── test_grpo_fruits_catcher.py
│   └── test_train_epoch.py
└── models/                   # Saved model files (generated)
    ├── grpo_fruits_catcher-*.pth
    └── ...
```

## 🔬 Research & Experimentation

This project serves as a testbed for:
- **Reinforcement Learning Algorithms**: GRPO (Group Relative Policy Optimization) and variants
- **Reward Engineering**: Different reward structures and clipping strategies  
- **Neural Architecture**: Hidden layer sizes, activation functions, regularization
- **Training Dynamics**: Learning rates, batch sizes, early stopping strategies
- **Game Mechanics**: Various game configurations and difficulty levels

The implementation demonstrates the core principles of GRPO as proposed by DeepSeek for their R1 model post-training, adapted for a simple game environment to make the algorithm more accessible and understandable.

## 📚 About GRPO

**Group Relative Policy Optimization (GRPO)** is a reinforcement learning algorithm developed by DeepSeek for post-training their DeepSeek-R1 model. This implementation adapts the core principles of GRPO to a simpler game environment, making it easier to understand and experiment with the algorithm.

This project builds upon the educational foundation laid by **Dr. Mihai Nica's Triangle Creatures implementation**, translating the concepts from JAX to PyTorch and from complex creature movement to simple fruit catching mechanics.

### Key GRPO Concepts Demonstrated:
- **Group-based Learning**: Training with batches of episodes for relative comparisons
- **Policy Optimization**: Direct optimization of policy parameters
- **Reward Processing**: Sophisticated reward shaping and normalization
- **Group Normalization**: Returns are normalized across the entire batch group for relative comparison
- **Stability Mechanisms**: Gradient clipping, entropy bonuses, and early stopping

### Educational Progression:
1. **Original DeepSeek Paper**: GRPO algorithm for language model post-training
2. **Dr. Mihai Nica's Video**: Triangle Creatures implementation in JAX 
3. **This Project**: Simplified fruits catching game in PyTorch

While this fruits catching game is much simpler than both language model post-training and triangle creature movement, it illustrates the fundamental mechanics of how GRPO works in a more accessible context. The implementation includes group normalization of returns (line 562-579 in `grpo_fruits_catcher.py`), which is essential for the relative policy optimization approach.

## 🤝 Contributing

This project is designed as an educational implementation of GRPO, inspired by Dr. Mihai Nica's Triangle Creatures work. Feel free to experiment with:

- Different reward functions in `grpo_fruits_catcher.py`
- New game mechanics or configurations
- Alternative neural network architectures
- Additional training algorithms
- Performance optimizations
- Educational improvements and documentation enhancements

**Educational Contributions Welcome:**
- Clearer explanations of GRPO concepts
- Additional visualization tools
- Comparison studies with other RL algorithms
- Tutorial content for beginners

**ML Expert Contributions Especially Needed:**
- Code review and validation of GRPO implementation
- Corrections to any ML domain inaccuracies
- Performance optimizations and best practices
- Academic accuracy improvements

When contributing, please maintain the educational focus and accessibility that makes this project valuable for learning GRPO concepts. Given that ~50% of the code was AI-assisted and the author is not an ML expert, domain expert review and corrections are particularly valuable.

## 📄 License

This project is provided as-is for educational and research purposes.

## 📚 References

1. **DeepSeek Team** - Original GRPO algorithm for DeepSeek-R1 post-training
2. **Dr. Mihai Nica** - "How does DeepSeek learn? GRPO explained with Triangle Creatures" 
   - YouTube: [https://www.youtube.com/watch?v=wXEvvg4YJ9I](https://www.youtube.com/watch?v=wXEvvg4YJ9I)
   - Original JAX implementation with Triangle Creatures
3. **This Project** - PyTorch adaptation with simplified fruits catching mechanics
   - ~50% AI-assisted development (GitHub Copilot Agent - Claude Sonnet 4 Preview)
   - Educational implementation by non-ML expert

---

## 📝 Documentation Note

This README consolidates and replaces the information previously found in:
- `TRAINING_ARGS.md` - Now integrated into the [Training Guide](#-training-guide) section
- `GAME_DEMO_ARGS.md` - Now integrated into the [Game Demo Guide](#-game-demo-guide) section

All command-line arguments, usage examples, and configuration options are now centralized in this single README file for easier navigation and maintenance.

---

**🍎 Happy Fruit Catching! 🤖**

*Inspired by Dr. Mihai Nica's educational work on GRPO*
