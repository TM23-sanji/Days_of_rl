# 🚀 Reinforcement Learning Algorithms Repository

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-orange.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2.2-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive collection of reinforcement learning algorithms implemented from scratch using PyTorch and Gymnasium. This repository serves as both a learning resource and a toolkit for experimenting with various RL approaches.

[Features](#-features) • [Quick Start](#-quick-start) • [Implemented Algorithms](#-implemented-algorithms) • [Environments](#-environments) • [Project Structure](#-project-structure) • [Planned Features](#-planned-features)

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Running Algorithms](#running-algorithms)
- [Implemented Algorithms](#-implemented-algorithms)
  - [Value-Based Methods](#value-based-methods)
  - [Policy-Based Methods](#policy-based-methods)
  - [Model-Based Methods](#model-based-methods)
- [Environments](#-environments)
- [Project Structure](#-project-structure)
- [Examples](#-examples)
- [Planned Features](#-planned-features)
- [Dependencies](#-dependencies)
- [Contributing](#-contributing)

---

## 🎯 About

This repository contains implementations of various reinforcement learning algorithms, ranging from classical methods to modern deep RL techniques. Each algorithm is implemented with clarity and educational value in mind, making it easier to understand the underlying principles of RL.

**Key Goals:**
- ✅ Implement core RL algorithms from scratch
- ✅ Provide clear, well-documented code
- ✅ Support multiple environments (discrete & continuous)
- ✅ Integrate experiment tracking with Weights & Biases
- 🔄 Expand with advanced algorithms (Actor-Critic, PPO, etc.)

---

## ✨ Features

- 🎮 **Multiple Environments**: CartPole, FrozenLake, Atari (Pong)
- 🧠 **Various Algorithms**: Value Iteration, Q-Learning, Cross-Entropy, DQN
- 📊 **Experiment Tracking**: Integrated Weights & Biases (wandb) support
- 🔧 **Custom Wrappers**: Environment preprocessing for Atari games
- 📝 **Clean Code**: Well-structured, type-hinted Python code
- 🎯 **Modular Design**: Easy to extend and experiment with

---

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd rl
   ```

2. **Install dependencies using `uv`** (recommended)
   ```bash
   uv sync
   ```

   Or using `pip`:
   ```bash
   pip install -e .
   ```

3. **Set up Weights & Biases** (optional but recommended)
   ```bash
   # Create a .env file in the project root
   echo "WANDB_API_KEY=your_api_key_here" > .env
   ```

### Running Algorithms

#### Cross-Entropy Method on CartPole
```bash
python envs/cartpole/cross_entropy.py
```

#### Cross-Entropy Method on FrozenLake
```bash
python envs/frozenlake/cross_entropy_naive.py
```

#### Deep Q-Network (DQN) on Pong
```bash
python envs/pong/dqn_pong.py --dev cuda  # Use GPU if available
python envs/pong/dqn_pong.py --dev cpu   # Use CPU
```

#### Q-Learning on FrozenLake
```bash
python envs/frozenlake/q_learning_ch6.py
```

#### Value Iteration on FrozenLake
```bash
python envs/frozenlake/v_iter.py
```

---

## 🧠 Implemented Algorithms

### Value-Based Methods

#### 1. **Value Iteration** 📊
- **Location**: `envs/frozenlake/v_iter.py`
- **Environment**: FrozenLake-v1
- **Type**: Model-based, tabular method
- **Description**: Dynamic programming algorithm that iteratively computes optimal value function
- **Status**: ✅ Implemented

#### 2. **Q-Learning** 🎯
- **Location**: `envs/frozenlake/q_learning_ch6.py`
- **Environment**: FrozenLake-v1
- **Type**: Model-free, off-policy, tabular method
- **Description**: Temporal difference learning algorithm for learning optimal Q-values
- **Status**: ✅ Implemented

#### 3. **Deep Q-Network (DQN)** 🧠
- **Location**: `envs/pong/dqn_pong.py`
- **Model**: `model/dqn_model.py`
- **Environment**: PongNoFrameskip-v4 (Atari)
- **Type**: Model-free, off-policy, deep RL
- **Features**:
  - Experience replay buffer
  - Target network for stable learning
  - Epsilon-greedy exploration
  - Custom Atari wrappers for preprocessing
- **Status**: ✅ Implemented

### Policy-Based Methods

#### 4. **Cross-Entropy Method** 🎲
- **Locations**: 
  - `envs/cartpole/cross_entropy.py`
  - `envs/frozenlake/cross_entropy_naive.py`
  - `envs/frozenlake/cross_entropy_non_slippery.py`
  - `envs/frozenlake/cross_entropy_work.py`
- **Environments**: CartPole-v1, FrozenLake-v1
- **Type**: Model-free, policy-based, on-policy
- **Description**: Gradient-free optimization method that selects top-performing episodes
- **Status**: ✅ Implemented

### Model-Based Methods

#### 5. **Q-Iteration** 🔄
- **Location**: `envs/frozenlake/q_iter.py`
- **Environment**: FrozenLake-v1
- **Type**: Model-based, tabular method
- **Description**: Dynamic programming approach for Q-value computation
- **Status**: ✅ Implemented

---

## 🎮 Environments

| Environment | Type | State Space | Action Space | Algorithms |
|------------|------|-------------|--------------|------------|
| **CartPole-v1** | Classic Control | Continuous (4D) | Discrete (2) | Cross-Entropy |
| **FrozenLake-v1** | Tabular | Discrete (16) | Discrete (4) | Value Iteration, Q-Learning, Q-Iteration, Cross-Entropy |
| **PongNoFrameskip-v4** | Atari | Image (210×160×3) | Discrete (6) | DQN |

---

## 📁 Project Structure

```
rl/
├── envs/                           # Environment-specific implementations
│   ├── cartpole/
│   │   └── cross_entropy.py       # Cross-entropy on CartPole
│   ├── frozenlake/
│   │   ├── cross_entropy_naive.py
│   │   ├── cross_entropy_non_slippery.py
│   │   ├── cross_entropy_work.py
│   │   ├── q_learning_ch6.py      # Q-Learning implementation
│   │   ├── q_iter.py              # Q-Iteration
│   │   └── v_iter.py              # Value Iteration
│   ├── pong/
│   │   └── dqn_pong.py            # DQN on Atari Pong
│   └── wrappers/
│       └── wrapper.py             # Custom environment wrappers
├── model/                          # Neural network architectures
│   └── dqn_model.py               # DQN CNN architecture
├── main.py                         # Main entry point (placeholder)
├── pyproject.toml                  # Project dependencies
├── uv.lock                         # Dependency lock file
└── README.md                       # This file
```

---

## 💡 Examples

### Example 1: Training DQN on Pong

```python
# Run with GPU acceleration
python envs/pong/dqn_pong.py --dev cuda --env PongNoFrameskip-v4
```

The algorithm will:
- Create an experience replay buffer
- Train using epsilon-greedy exploration
- Log metrics to Weights & Biases
- Save the best model checkpoint

### Example 2: Cross-Entropy on CartPole

```python
python envs/cartpole/cross_entropy.py
```

This will:
- Train a policy network using the cross-entropy method
- Filter top-performing episodes (percentile-based)
- Track training metrics via wandb
- Stop when mean reward exceeds threshold

---

## 🔮 Planned Features

### Algorithms to Implement

- [ ] **Actor-Critic Methods**
  - [ ] Advantage Actor-Critic (A2C)
  - [ ] Asynchronous Advantage Actor-Critic (A3C)
  
- [ ] **Policy Gradient Methods**
  - [ ] REINFORCE
  - [ ] Proximal Policy Optimization (PPO)
  - [ ] Trust Region Policy Optimization (TRPO)

- [ ] **Advanced Q-Learning**
  - [ ] Double DQN
  - [ ] Dueling DQN
  - [ ] Prioritized Experience Replay
  - [ ] Rainbow DQN

- [ ] **Model-Based RL**
  - [ ] Dyna-Q
  - [ ] Model-Agnostic Meta-Learning (MAML)

- [ ] **Continuous Control**
  - [ ] Deep Deterministic Policy Gradient (DDPG)
  - [ ] Twin Delayed DDPG (TD3)
  - [ ] Soft Actor-Critic (SAC)

### Infrastructure Improvements

- [ ] Unified training/evaluation framework
- [ ] Config files (YAML/JSON) for hyperparameters
- [ ] More comprehensive logging and visualization
- [ ] Unit tests for all algorithms
- [ ] CI/CD pipeline
- [ ] Documentation with Sphinx/ReadTheDocs
- [ ] Performance benchmarks and comparisons

---

## 📦 Dependencies

Core dependencies are defined in `pyproject.toml`:

- **PyTorch** (≥2.9.1): Deep learning framework
- **Gymnasium** (≥1.2.2): Standard RL environments
- **NumPy** (≥2.3.5): Numerical computations
- **Weights & Biases** (≥0.23.0): Experiment tracking
- **Stable-Baselines3** (≥2.7.0): RL utilities and wrappers
- **python-dotenv** (≥0.9.9): Environment variable management

For GPU support, install PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Implement new algorithms** - Check the [Planned Features](#-planned-features) section
2. **Fix bugs** - Report issues or submit pull requests
3. **Improve documentation** - Make the code more accessible
4. **Add tests** - Help ensure code quality
5. **Optimize performance** - Speed up training or reduce memory usage

### Guidelines

- Follow existing code style and structure
- Add docstrings to new functions/classes
- Update this README when adding new algorithms
- Use type hints for better code clarity
- Test your changes before submitting

---

## 📊 Experiment Tracking

All algorithms integrate with **Weights & Biases** for experiment tracking. Metrics logged include:

- Training rewards/episode returns
- Loss values
- Exploration rates (epsilon)
- Learning curves
- Hyperparameters

View your experiments at: [wandb.ai](https://wandb.ai)

---

## 📚 Learning Resources

This repository is designed as a learning tool. Recommended reading:

- **Sutton & Barto** - Reinforcement Learning: An Introduction
- **Deep RL Bootcamp** - CS294-112 at UC Berkeley
- **Spinning Up in Deep RL** - OpenAI's educational resource

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- OpenAI Gym/Gymnasium team for the excellent environments
- PyTorch team for the deep learning framework
- Weights & Biases for experiment tracking tools

---

<div align="center">

**Happy Learning! 🎓**

*Star ⭐ this repo if you find it helpful!*

</div>
