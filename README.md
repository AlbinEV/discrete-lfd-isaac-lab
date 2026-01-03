# Discrete LfD for Robotic Polishing

Learning from Demonstration (LfD) environment with **discrete action space** for Isaac Lab 2.0.1.

## 📦 What's Included

- `robo_pp_lfd_discrete/` - Complete environment package with:
  - Discrete action space (27 actions: 3³ combinations of dx,dy,dz)
  - Sequential axis reward function (X→Y→Z tracking)
  - Differential IK controller
  - PPO agent configuration
- `scripts/` - Training, evaluation, and visualization tools
- `examples/` - Demo trajectories (train: 1k waypoints, test: 3k waypoints)

## 🚀 Quick Start

### 1. Prerequisites

- [Isaac Lab 2.0.1](https://github.com/isaac-sim/IsaacLab) installed and working
- Python 3.10
- CUDA 11.8+ (for GPU acceleration)

### 2. Installation

Copy the environment package into your Isaac Lab installation:

```bash
# Navigate to your Isaac Lab tasks directory
cd <ISAAC_LAB_PATH>/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/

# Copy the environment package
cp -r /path/to/robo_pp_lfd_discrete ./

# Add to __init__.py (if not auto-registered)
# In omni/isaac/lab_tasks/manager_based/__init__.py:
# from .robo_pp_lfd_discrete import *
```

### 3. Install Dependencies

```bash
# In your Isaac Lab environment
pip install rl-games plotly
```

### 4. Train

```bash
cd <ISAAC_LAB_PATH>
./isaaclab.sh -p /path/to/scripts/train.py --num_envs 64 --max_iterations 500
```

**Training takes ~2-3 hours** on RTX 4090 (64 parallel environments).

### 5. Evaluate

```bash
./isaaclab.sh -p /path/to/scripts/play.py \
    --checkpoint logs/rl_games/.../nn/last_<checkpoint>.pth \
    --trajectory examples/traj_2_shifted.json \
    --num_episodes 100 \
    --save_stats results.json
```

### 6. Visualize Trajectory

```bash
python scripts/visualize_trajectory.py examples/traj_1_shifted.json -o traj_1.html
# Open traj_1.html in browser
```

## 📊 Environment Details

### Action Space
- **Type**: Discrete(27)
- **Encoding**: 27 = 3³ combinations of (dx, dy, dz) ∈ {-1, 0, +1}
- **Step size**: 0.161 cm (configurable via `LFD_DIS_STEP_CM`)

### Observation Space
- **Shape**: (64, 36) - 64 timesteps × 36 features
- **Features**: joint positions (7), velocities (7), accelerations (7), torques (7), 
  end-effector position (3), goal position (3), force_z (1), phase (1)

### Reward Function (v4)
Sequential axis tracking with four components:
1. **Waypoint distance** (w=1.0): Gaussian reward on active axis only
2. **Progress** (w=2.0): Bonus for reducing distance  
3. **Hit** (w=5.0): Bonus when all axes within tolerance (ε=0.01m)
4. **Time penalty** (w=-0.1): Encourages faster completion

Axis sequence: X → Y → Z (unlocked when previous axis within tolerance)

### Hyperparameters
```python
# Reward tuning
eps_in = 0.01      # Hit tolerance (m)
sigma = 0.03       # Gaussian spread (m)
c_clip = 0.02      # Clipping value

# IK controller
Kp = 129.78        # Proportional gain
Kd = 1.63          # Derivative gain
```

## 🗂️ File Structure

```
.
├── robo_pp_lfd_discrete/          # Environment package
│   ├── Polish_Env_LfD_Discrete.py # Main environment
│   ├── __init__.py                # Registration
│   ├── agents/                    # RL agent configs
│   │   └── rl_games_ppo_cfg.yaml
│   └── rewards/                   # Reward functions
│       ├── __init__.py
│       └── reward_wp_v4_dis.py    # Sequential axis reward
├── scripts/                       # Utilities
│   ├── train.py                   # Training script
│   ├── play.py                    # Evaluation script
│   └── visualize_trajectory.py    # HTML visualization
├── examples/                      # Demo data
│   ├── traj_1_shifted.json        # Training trajectory (1k wpts)
│   └── traj_2_shifted.json        # Test trajectory (3k wpts)
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## 🔧 Configuration

Environment behavior is controlled via environment variables:

```bash
# Trajectory
export LFD_TRAJ_PATH="/path/to/trajectory.json"
export LFD_TRAJ_START_IDX=0          # Start waypoint index
export LFD_TRAJ_WPT_NOISE=0.005      # Waypoint noise (m)

# Action space
export LFD_DIS_STEP_CM=0.161         # Discrete step size (cm)

# IK controller  
export LFD_IK_KP=129.78              # Proportional gain
export LFD_IK_KD=1.63                # Derivative gain

# Reward weights
export LFD_REWARD_W_WP=1.0           # Waypoint distance weight
export LFD_REWARD_W_PROG=2.0         # Progress weight
export LFD_REWARD_W_HIT=5.0          # Hit bonus weight
export LFD_REWARD_W_TIME=-0.1        # Time penalty weight

# Reward parameters
export LFD_REWARD_EPS_IN=0.01        # Hit tolerance (m)
export LFD_REWARD_SIGMA=0.03         # Gaussian spread (m)
export LFD_REWARD_C_CLIP=0.02        # Clipping value
```

## 📈 Expected Results

With provided hyperparameters (500 iterations, 64 envs):

- **Training trajectory** (1k waypoints): ~95% success rate
- **Test trajectory** (3k waypoints, unseen): ~40% success rate
- **Generalization**: Policy handles 3× longer trajectories without retraining

Sample results on test trajectory:
```
Episodes: 659
Success rate: 39.91%
Mean waypoint coverage: 92.47%
Mean steps: 5881
```

## 📝 Citation

If you use this environment in your research, please cite:

```bibtex
@software{vtt_rl_discrete_lfd_2026,
  title = {Discrete Action Space Learning from Demonstration for Robotic Polishing},
  author = {VTT Technical Research Centre of Finland},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/repo},
  note = {Developed in collaboration with VTT (Technical Research Centre of Finland), Oulu, Finland}
}
```

## 🏢 Acknowledgements

This work was developed in collaboration with **VTT Technical Research Centre of Finland**, one of Europe's leading research organizations with over 2,300 professionals advancing sustainable solutions through science and technology.

**VTT Oulu** is one of VTT's six research locations in Finland, specializing in intelligent production, robotics, and industrial automation research.

More information: [www.vttresearch.com](https://www.vttresearch.com)

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🤝 Contributing

This is a research prototype. For questions or issues:
1. Check Isaac Lab 2.0.1 documentation
2. Verify environment variables are set correctly
3. Ensure dependencies (rl-games, plotly) are installed

## 🔗 Related

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) - Robot learning framework
- [RL-Games](https://github.com/Denys88/rl_games) - RL training library
- Paper: *(add publication link when available)*
