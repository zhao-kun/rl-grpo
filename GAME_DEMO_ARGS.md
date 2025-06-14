# 🎮 Run Game Demo - Command Line Arguments

## 🚀 Quick Start

### Default Usage
```bash
python run_game_demo.py
```
Automatically finds the latest trained model and runs the game with full instructions.

## 📋 All Available Arguments

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

## 🎯 Usage Examples

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
