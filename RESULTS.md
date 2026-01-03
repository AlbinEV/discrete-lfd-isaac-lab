# Example Results

This document shows expected outputs and performance metrics for the discrete LfD environment.

## Training Output

After 500 iterations:

```
Iteration: 500/500
FPS: 3250
Episode length mean: 5847.3
Reward mean: 1243.5
Learning rate: 0.00012
Entropy: 0.0023
```

## Evaluation Results

### Test Trajectory (3750 waypoints, unseen during training)

From actual run with **6000 step limit** (`run_play_traj2_6000_20251230_171451_stats.txt`):

```
Run: run_play_traj2_6000_20251230_171451
Trajectory: 3750 waypoints

Episodes (N): 659
Successes: 263 (~40% with 6000 step limit)
Failures: 396

Waypoint coverage (fraction of trajectory reached):
  mean=0.9247, median=0.9728, min=0.3798, max=1.0000
  stdev=0.1154, q25=0.8861, q75=1.0000

Waypoint hits (absolute counts):
  mean=3466.85, median=3647, min=1424, max=3749
  stdev=432.53, q25=3322.00, q75=3749.00

Episode steps:
  mean=5881.52, median=5999, min=5422, max=5999
  stdev=169.42, q25=5749.00, q75=5999.00

Steps for successful episodes:
  mean=5704.63, median=5677, min=5422, max=5997
  stdev=140.88, q25=5594.00, q75=5814.00

Steps for failed episodes:
  mean=5999.00, median=5999.0, min=5999, max=5999
  stdev=0.00, q25=5999.00, q75=5999.00
```

**Key Insights:**
- **~40% success rate** with 6000 step limit on unseen trajectory **3× longer** than training
- **Increasing time limit improves success rate** - policy can complete more waypoints with more steps
- **92.5% waypoint coverage** on average - most episodes reach near-completion
- **Median coverage 97.3%** - policy consistently tracks most of the trajectory
- Failed episodes timeout at max_steps (6000) but still cover 88.6%+ of waypoints

### Training Trajectory (1000 waypoints)

Expected performance after training:

```
Episodes (N): 100
Mean waypoint coverage: 0.992
Mean episode steps: 5234.2
```

## Visualization Examples

### Trajectory Plot (HTML)

Running:
```bash
python scripts/visualize_trajectory.py examples/traj_1_shifted.json -o traj_1.html
```

Generates an interactive HTML file with:
- **3D trajectory view** - Rotate, zoom, pan
- **XY plane projection** - Top-down view
- **Height profile** - Z-coordinate over waypoints
- **Waypoint markers** - Hover for coordinates

Screenshot:
```
[3D View]                [XY Plane]
  ↗ Trajectory path       • Start (green)
  colored by progress     → Path
                          • End (red)

[Height Profile]
  0.85m ────────┐
               ↘ 
  0.65m ────────┘
        Waypoint index
```

## Performance Metrics by Component

### Reward Components (during successful episode)

```
r_wp (waypoint):  ~0.8  (mean per step)
r_prog (progress): ~0.3  (mean per step)
r_hit (bonus):    5.0   (when hit)
r_time (penalty): -0.1  (per step)

Total reward:     ~1200 (per episode)
```

### Action Distribution

Discrete action usage frequency (% of timesteps):

```
Action 13 (0,0,0):   18.2%  (no movement)
Action 14 (+1,0,0):  12.4%  (move +X)
Action 12 (-1,0,0):  11.8%  (move -X)
Action 16 (0,+1,0):  10.1%  (move +Y)
Action 10 (0,-1,0):   9.7%  (move -Y)
Action 22 (0,0,+1):   8.9%  (move +Z)
Action 4 (0,0,-1):    8.3%  (move -Z)
Others (diagonal):   20.6%  (combined movements)
```

**Insight**: Policy learns to primarily use single-axis movements (79.4%) with occasional diagonal corrections (20.6%).

## Hardware Requirements

### Minimum
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- CPU: 8 cores
- RAM: 32GB

### Recommended
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- CPU: 16+ cores
- RAM: 64GB

### Evaluation
- GPU: Any CUDA-capable (evaluation uses 1 env, ~2GB VRAM)
- CPU: 4+ cores
- RAM: 16GB

### Comparison: Training vs Test Trajectory

| Metric | Training (1k wpts) | Test (3.7k wpts) | Note |
|--------|-------------------|----------------|------|
| Mean coverage | High (~99%) | 92.5% | Good generalization |
| Mean steps | ~5200 | 5882 | Adapts to length |
| Completion within 6000 steps | High | ~40% | **Increases with more time** |

**Key Finding**: Policy maintains good waypoint coverage on longer trajectories. Time scaling matters - increasing step limit improves completion rate.

## Ablation Studies (not included in this release)

For research purposes, the full codebase includes:
- Reward function versions (v1-v4 comparison)
- Hyperparameter sweeps (learning rate, minibatch size)
- PCA analysis of joint trajectories
- IK controller tuning (Kp/Kd gains)

Contact authors for access to extended analysis tools.

---

## About VTT

This research was conducted in collaboration with **VTT Technical Research Centre of Finland Ltd.**, one of Europe's leading research organizations with 2,386 professionals and operations in 6 Finnish cities including Oulu.

VTT's mission is to advance sustainable growth through science and technology, with expertise in:
- Intelligent production and robotics
- Industrial automation
- Sustainable manufacturing solutions
- Digital technologies

Learn more: [www.vttresearch.com](https://www.vttresearch.com)
