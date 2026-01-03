#!/usr/bin/env python3
"""Minimal play/evaluation script for discrete LfD environment."""

import argparse
import os
import json
from omni.isaac.lab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate discrete LfD policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
parser.add_argument("--trajectory", type=str, default="examples/traj_2_shifted.json", 
                    help="Path to trajectory file")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of evaluation episodes")
parser.add_argument("--save_stats", type=str, default=None, help="Path to save statistics JSON")
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching
import torch
import numpy as np
from omni.isaac.lab_tasks.utils import parse_env_cfg
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

def play():
    """Evaluate the discrete LfD policy."""
    
    # Environment configuration
    env_cfg = parse_env_cfg("Robo_PP-LfD-Discrete-v0", num_envs=1, use_fabric=True)
    
    # Set trajectory path
    traj_path = os.path.abspath(args_cli.trajectory)
    os.environ["LFD_TRAJ_PATH"] = traj_path
    
    # Create environment
    env = env_cfg.env_class(cfg=env_cfg, render=True)
    
    # Load checkpoint
    checkpoint = torch.load(args_cli.checkpoint)
    
    # Statistics
    stats = {
        "episodes": [],
        "success_rate": 0.0,
        "mean_waypoint_coverage": 0.0,
        "mean_steps": 0.0
    }
    
    # Run episodes
    obs = env.reset()
    for episode in range(args_cli.num_episodes):
        done = False
        steps = 0
        episode_stats = {"waypoints_hit": 0, "steps": 0, "success": False}
        
        while not done and steps < 6000:
            # Get action from policy
            with torch.no_grad():
                action = checkpoint["model"].act(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if "waypoint_idx" in info:
                episode_stats["waypoints_hit"] = info["waypoint_idx"][0].item()
        
        episode_stats["steps"] = steps
        episode_stats["success"] = info.get("success", [False])[0]
        stats["episodes"].append(episode_stats)
        
        print(f"Episode {episode+1}/{args_cli.num_episodes}: "
              f"Steps={steps}, Waypoints={episode_stats['waypoints_hit']}, "
              f"Success={episode_stats['success']}")
        
        obs = env.reset()
    
    # Compute summary statistics
    successes = sum(1 for ep in stats["episodes"] if ep["success"])
    stats["success_rate"] = successes / args_cli.num_episodes
    stats["mean_steps"] = np.mean([ep["steps"] for ep in stats["episodes"]])
    stats["mean_waypoint_coverage"] = np.mean(
        [ep["waypoints_hit"] for ep in stats["episodes"]]
    )
    
    print(f"\n=== Evaluation Summary ===")
    print(f"Success Rate: {stats['success_rate']*100:.2f}%")
    print(f"Mean Steps: {stats['mean_steps']:.1f}")
    print(f"Mean Waypoint Coverage: {stats['mean_waypoint_coverage']:.1f}")
    
    # Save statistics
    if args_cli.save_stats:
        with open(args_cli.save_stats, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to {args_cli.save_stats}")
    
    env.close()

if __name__ == "__main__":
    play()
    simulation_app.close()
