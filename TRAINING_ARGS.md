# 🍎 Fruits Catcher GRPO Training - Command Line Arguments

## 🚀 Quick Start

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

#### � Early Stopping Control
```bash
# Quick testing with early stopping after 50 epochs
python main.py --total-epochs 1000 --patience 50

# Conservative training with longer patience
python main.py --total-epochs 5000 --patience 500

# Aggressive early stopping for quick experiments
python main.py --total-epochs 2000 --patience 100
```

#### � CPU Training
```bash
python main.py --device cpu --batch-size 8 --total-epochs 500
```

#### �📂 Custom Model Name
```bash
python main.py --model-name my_custom_model --total-epochs 1500
```

## 📋 All Available Arguments

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
