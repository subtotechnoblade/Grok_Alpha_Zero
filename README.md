# AlphaZero General Implementation & Enhancements
A from scratch, high performance, implementation of DeepMind's Alpha Zero system in Tensorflow, featuring significant architectural and algorithmic improvements for efficiency, scalability, and performance. This project has been a long time dream of mine to complete.

To Henry, thanks for suggesting Alpha Zero to me in grade 10, I would never have come this far. Thank you for showing me my passion and my dream in life. Programming and ML has been the greatest passion I could have ever asked for. It has been a while but I'm nearing completion, I hope this shows you how far I've come from where I started. Again, thank you for showing me my path.

## 🚀 Key Features & Enhancements
This implementation includes the core AlphaZero algorithm but is significantly enhanced for performance and learning efficiency:


## 🎯 Core Implementation
General Game Framework: Framework capable of learning multiple perfect information games from scratch through self-play.

Monte Carlo Tree Search (MCTS): The core planning algorithm implemented as an implicit tree with memory efficiency in mind.

Residual Networks (ResNet): Standard deep neural network architecture for stable learning of complex value and policy functions.

## ⚡ Performance & Scalability
High-Performance Inference Server: A custom inference server that batches policy/value network requests from multiple MCTS workers. This is drastically more efficient than spawning multiple ONNX runtime instances, minimizing GPU overhead and maximizing throughput.

Multi-worker Self-Play: Parallel self-play data generation utilizing multiple CPU cores, significantly accelerating training.

Multi-Platform GPU Acceleration: Model exported to ONNX format for fast inference across multiple execution providers:

- TensorRT (NVIDIA GPU - Optimized)
- CUDA (NVIDIA GPU - Standard)
- DirectML (GPU - Windows)
- CPU (Fallback)

## 🧩 Implementation Checklist

### ✅ Completed Features

#### Core Algorithm & MCTS
- [x] Implicit MCTS Tree Representation (Memory-Optimized)
- [x] MCTS Node Pruning
- [x] Hybrid Training Target Z * 0.75 + Q * 0.25

#### Performance & Optimization
- [x] Numba JIT Acceleration for Critical Functions
- [x] Multi-Platform ONNX Inference (TensorRT, CUDA, DirectML, CPU)
- [x] Batch Inference Server for Parallel Workers
- [x] Neural Network Output Caching

#### Infrastructure & Data Management
- [x] Game Compatibility Test Suite
- [x] HDF5-Based Experience Replay Buffer

#### Network Architecture
- [x] Residual Network (ResNet) Blocks

#### Training Stabilization Techniques
- [x] GrokFast Optimization
- [x] OrthoGrad Regularization
- [x] StableMax Activation

### 🔄 In Progress / Planned

#### Parallelization & Gameplay
- [ ] Virtual Loss for Parallel MCTS (Multi-agent Competition)
- [ ] Rich Representation from: https://arxiv.org/abs/2310.06110

## 🕹️ Implemented Games

| Game                 | 	Status | Description                                 |
|----------------------|---------|---------------------------------------------|
| Tic-Tac-Toe          | 	✅      | Complete	Simple validation environment      |
| Connect4	            | ✅       | Complete	Classic 6x7 board game             |
| Gomoku	              | ✅       | Complete	5-in-a-row on a 15x15 board        |
| Ultimate Tic-Tac-Toe | 	🔄     | In Progress	Complex, strategic variant      |
| Chopsticks	          | 🔄      | In Progress	Simple perfect-information game |

# 🏗️ Project Architecture
```mermaid
graph TD
    A[Self-Play Workers CPU xN] -->|Send Batches| B[Inference Server];
    B -->|Uses| C[ONNX Model];
    C -->|Runs on| D{TensorRT / CUDA / DML / CPU};
    B -->|Returns Policy/Value| A;
    A -->|Generates| E[Game Data, stored in h5py];
    E -->|Processed by| F[Data Preprocessing]; 
    F --> G[Training Loop];
    G -->|Updates| H[TensorFlow Model];
    H -->|Exported to| C;
```
## 🚦 Getting Started

### Installation & Training
Clone the repo:
```bash
git clone https://github.com/subtotechnoblade/Grok_Alpha_Zero
cd Grok_Alpha_Zero
```

### Install dependencies:
For GPU training:
```bash
pip install -r requirements-training-gpu.txt
```

For CPU training:
```bash
pip install -r requirements-training-cpu.txt
```

### Train Alpha Zero for Connect4
```bash
cd Connect4
python main.py
```

## Issues
- numba unable to find the correct cache key. Solution: Delete pycache and restart training. A fix will be introduced later


## References & Acknowledgements

### Core Algorithm Papers
- **AlphaZero**: Silver, D., et al. (2017, December 5). Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm. https://arxiv.org/abs/1712.01815
- **MuZero**: Schrittwieser, J., et al. (2020, February 21). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. https://arxiv.org/abs/1911.08265
- **Gumbel AlphaZero/MuZero**: Danihelka, I., et al. (2023, February 13). Policy improvement by planning with Gumbel. OpenReview. https://openreview.net/forum?id=bERaNdoegnO

### Network Architecture  
- **Residual Networks**: He, K., et al. (2015, December 10). Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385
- **GARB Networks with attention and Alpha Zero training techniques**: Gao, Yifan & Wu, Lezhou. (2021). Efficiently Mastering the Game of NoGo with Deep Reinforcement Learning Supported by Domain Knowledge. Electronics. 10. 1533. 10.3390/electronics10131533. 

### Grokking Techniques
- **Grokfast**: Lee, J., et al. (2024, June 5). Grokfast: Accelerated Grokking by Amplifying Slow Gradients. https://arxiv.org/abs/2405.20233
- **Orthograd & Stablemax**: Prieto, L. et al. (2025, May 19). Grokking at the Edge of Numerical Stability. https://arxiv.org/abs/2501.04697

