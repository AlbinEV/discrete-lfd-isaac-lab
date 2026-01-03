# Installation Guide

Step-by-step instructions to integrate this environment into Isaac Lab 2.0.1.

## Prerequisites

- **Isaac Lab 2.0.1** installed and tested
- **Python 3.10** (bundled with Isaac Lab)
- **CUDA 11.8+** for GPU acceleration
- **Git** for cloning this repository

## Installation Steps

### 1. Clone This Repository

```bash
cd ~/workspace  # or your preferred location
git clone https://github.com/yourusername/discrete-lfd-isaac-lab.git
cd discrete-lfd-isaac-lab
```

### 2. Locate Isaac Lab Tasks Directory

Find your Isaac Lab installation and navigate to the tasks directory:

```bash
# Example path - adjust to your installation
export ISAAC_LAB_PATH=~/IsaacLab
cd $ISAAC_LAB_PATH/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/
```

### 3. Copy Environment Package

Copy the `robo_pp_lfd_discrete` package into the tasks directory:

```bash
cp -r ~/workspace/discrete-lfd-isaac-lab/robo_pp_lfd_discrete ./
```

Your directory structure should now look like:
```
omni/isaac/lab_tasks/manager_based/
├── classic/
├── locomotion/
├── manipulation/
├── robo_pp_lfd_discrete/  # <-- New package
│   ├── Polish_Env_LfD_Discrete.py
│   ├── __init__.py
│   ├── agents/
│   └── rewards/
└── __init__.py
```

### 4. Register Environment

**Option A: Auto-registration (Isaac Lab 2.0.1+)**

If Isaac Lab auto-discovers environments, you may skip this step.

**Option B: Manual registration**

Edit `omni/isaac/lab_tasks/manager_based/__init__.py`:

```python
# Add at the end of the file
from .robo_pp_lfd_discrete import *
```

### 5. Install Python Dependencies

Activate Isaac Lab's Python environment and install dependencies:

```bash
cd $ISAAC_LAB_PATH

# Install RL-Games (PPO training library)
./isaaclab.sh -p -m pip install rl-games

# Install Plotly (visualization)
./isaaclab.sh -p -m pip install plotly
```

### 6. Verify Installation

Test that the environment is registered:

```bash
cd $ISAAC_LAB_PATH
./isaaclab.sh -p ~/workspace/discrete-lfd-isaac-lab/scripts/train.py --help
```

If successful, you should see the training script's help message.

## Troubleshooting

### "Environment not found"

**Problem**: Isaac Lab cannot find the `Robo_PP-LfD-Discrete-v0` environment.

**Solution**:
1. Verify the package is in the correct directory: `omni/isaac/lab_tasks/manager_based/robo_pp_lfd_discrete/`
2. Check `__init__.py` contains environment registration
3. Restart any running Isaac Lab processes

### "No module named 'rl_games'"

**Problem**: RL-Games library not installed.

**Solution**:
```bash
cd $ISAAC_LAB_PATH
./isaaclab.sh -p -m pip install rl-games
```

### "Trajectory file not found"

**Problem**: Environment cannot locate trajectory JSON file.

**Solution**:
Set the `LFD_TRAJ_PATH` environment variable:
```bash
export LFD_TRAJ_PATH="/absolute/path/to/traj_1_shifted.json"
```

Or modify the script to use absolute paths.

### "CUDA out of memory"

**Problem**: GPU runs out of memory during training.

**Solution**:
Reduce the number of parallel environments:
```bash
./isaaclab.sh -p scripts/train.py --num_envs 32  # instead of 64
```

### Import errors from robo_pp_lfd

**Problem**: Environment references missing dependencies from `robo_pp_lfd` package.

**Solution**:
This package is **standalone** and should not import from `robo_pp_lfd`. If you see such errors:
1. Check you're using the correct environment class: `Polish_Env_LfD_Discrete`
2. Verify all files in `robo_pp_lfd_discrete/` are present
3. Ensure `__pycache__` is clean: `find . -type d -name __pycache__ -exec rm -rf {} +`

## Next Steps

Once installed, proceed to:
- [Training Guide](README.md#4-train) in main README
- [Configuration Options](README.md#-configuration) for environment customization
- Run visualization: `python scripts/visualize_trajectory.py examples/traj_1_shifted.json`

## File Locations Summary

After installation:

| File Type | Location |
|-----------|----------|
| Environment package | `$ISAAC_LAB_PATH/.../manager_based/robo_pp_lfd_discrete/` |
| Training script | `~/workspace/discrete-lfd-isaac-lab/scripts/train.py` |
| Demo trajectories | `~/workspace/discrete-lfd-isaac-lab/examples/*.json` |
| Checkpoints (after training) | `$ISAAC_LAB_PATH/logs/rl_games/...` |

## Uninstallation

To remove the environment:

```bash
cd $ISAAC_LAB_PATH/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/
rm -rf robo_pp_lfd_discrete/
```

And remove the import line from `__init__.py` if added manually.
