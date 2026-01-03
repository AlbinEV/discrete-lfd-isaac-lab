#!/usr/bin/env python3
"""Minimal training script for discrete LfD environment."""

import argparse
from omni.isaac.lab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train discrete LfD policy")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--max_iterations", type=int, default=500, help="Maximum training iterations")
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching
import os
import torch
from omni.isaac.lab_tasks.utils import parse_env_cfg
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

def train():
    """Train the discrete LfD policy."""
    
    # Environment configuration
    env_cfg = parse_env_cfg(
        "Robo_PP-LfD-Discrete-v0",
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.headless
    )
    
    # Set trajectory path (modify as needed)
    os.environ["LFD_TRAJ_PATH"] = os.path.join(
        os.path.dirname(__file__), "..", "examples", "traj_1_shifted.json"
    )
    
    # RL-Games configuration
    config = {
        "params": {
            "seed": 42,
            "algo": {
                "name": "a2c_continuous"
            },
            "model": {
                "name": "continuous_a2c_logstd"
            },
            "network": {
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": 0},
                        "fixed_sigma": True
                    }
                },
                "mlp": {
                    "units": [256, 128, 64],
                    "activation": "elu",
                    "d2rl": False,
                    "initializer": {"name": "default"},
                    "regularizer": {"name": "None"}
                }
            },
            "load_checkpoint": False,
            "config": {
                "name": "Robo_PP_LfD_Discrete",
                "env_name": "rlgpu",
                "multi_gpu": False,
                "ppo": True,
                "mixed_precision": False,
                "normalize_input": True,
                "normalize_value": True,
                "value_bootstrap": True,
                "num_actors": args_cli.num_envs,
                "reward_shaper": {"scale_value": 1.0},
                "normalize_advantage": True,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 3e-4,
                "lr_schedule": "adaptive",
                "kl_threshold": 0.008,
                "score_to_win": 1000000,
                "max_epochs": args_cli.max_iterations,
                "save_best_after": 50,
                "save_frequency": 50,
                "grad_norm": 1.0,
                "entropy_coef": 0.0,
                "truncate_grads": True,
                "e_clip": 0.2,
                "horizon_length": 64,
                "minibatch_size": 64,
                "mini_epochs": 5,
                "critic_coef": 2,
                "clip_value": True,
                "seq_len": 4,
                "bounds_loss_coef": 0.0001,
                "player": {
                    "deterministic": False,
                    "games_num": 10,
                    "print_stats": True
                }
            }
        }
    }
    
    # Register environment
    vecenv.register(
        "rlgpu",
        lambda config_name, num_actors, **kwargs: env_cfg.env_class(cfg=env_cfg, render=not args_cli.headless)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "rlgpu", "env_creator": lambda **kwargs: None})
    
    # Train
    runner = Runner()
    runner.load(config)
    runner.reset()
    runner.run({"train": True, "play": False})

if __name__ == "__main__":
    train()
    simulation_app.close()
