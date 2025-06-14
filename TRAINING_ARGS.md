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
  --hidden-size 1024 \
  --batch-size 32 \
  --total-epochs 3000 \
  --lr-rate 5e-5 \
  --max-steps 200
```

#### 💻 CPU Training
```bash
python main.py --device cpu --batch-size 8 --total-epochs 500
```

#### 📂 Custom Model Name
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
- `--hidden-size` - Neural network hidden layer size (default: 768)
- `--batch-size` - Training batch size (default: 16)
- `--total-epochs` - Total training epochs (default: 2000)
- `--max-steps` - Maximum steps per episode (default: 150)
- `--lr-rate` - Learning rate (default: 1e-4)
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

## 📊 Example Training Configurations

### Beginner (Fast Training)
```bash
python main.py --total-epochs 500 --batch-size 8 --hidden-size 256
```

### Intermediate (Balanced)
```bash  
python main.py --total-epochs 1500 --batch-size 16 --hidden-size 512 --compile
```

### Advanced (High Performance)
```bash
python main.py --total-epochs 3000 --batch-size 32 --hidden-size 1024 --compile --lr-rate 5e-5
```

### Research (Long Training)
```bash
python main.py --total-epochs 5000 --batch-size 64 --hidden-size 2048 --max-steps 300 --compile
```
