"""
Waypoint reward v4 (discrete actions) - Sequential axis tracking (X -> Y -> Z) with time penalty.

Created: 2025-12-22
Author: VTT_RL Project
Status: Experimental
Based on: reward_wp_v3.py

Shaping (axis-specific):
    active_dist = |dx| if |dx| >= eps_in else |dy| if |dy| >= eps_in else |dz|
    r_wp   = exp(-(active_dist^2) / sigma^2)
    r_prog = clamp(prev_dist - active_dist, [-c_clip, c_clip])
    r_hit  = 1.0 if |dx|<eps_in and |dy|<eps_in and |dz|<eps_in else 0.0
    r_time = 1.0 (cost = w_time * r_time, w_time < 0)
    reward = w_wp * r_wp + w_prog * r_prog + w_hit * r_hit + w_time * r_time
"""

from __future__ import annotations

import torch


def waypoint_reward_v4_dis(
    ee_pos: torch.Tensor,
    waypoints: torch.Tensor,
    wpt_idx: torch.Tensor,
    prev_dist: torch.Tensor,
    *,
    eps_in: float = 0.01,
    sigma: float = 0.03,
    c_clip: float = 0.02,
    w_wp: float = 1.0,
    w_prog: float = 2.0,
    w_hit: float = 5.0,
    w_time: float = -0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute waypoint reward v4 (discrete actions, sequential axis, with time penalty).

    Returns:
        reward: [N] total reward.
        new_wpt_idx: [N] updated waypoint indices.
        new_prev_dist: [N] updated distance buffer (active axis).
        hit: [N] bool tensor indicating waypoint reached (all axes within eps_in).
    """
    device = ee_pos.device
    N = ee_pos.shape[0]
    env_ids = torch.arange(N, device=device)

    goal = waypoints[env_ids, wpt_idx]  # [N,3]
    delta = torch.abs(ee_pos - goal)    # [N,3]

    # Determine active axis: X first, then Y, then Z
    axis_x = delta[:, 0] >= eps_in
    axis_y = (~axis_x) & (delta[:, 1] >= eps_in)
    active_dist = torch.where(
        axis_x,
        delta[:, 0],
        torch.where(axis_y, delta[:, 1], delta[:, 2]),
    )

    hit = (delta[:, 0] < eps_in) & (delta[:, 1] < eps_in) & (delta[:, 2] < eps_in)
    r_hit = hit.float()

    r_wp = torch.exp(-((active_dist * active_dist) / (sigma * sigma)))
    r_prog = torch.clamp(prev_dist - active_dist, min=-c_clip, max=c_clip)
    r_time = torch.ones(N, device=device)

    reward = w_wp * r_wp + w_prog * r_prog + w_hit * r_hit + w_time * r_time

    max_idx = torch.tensor(waypoints.shape[1] - 1, device=device, dtype=wpt_idx.dtype)
    new_wpt_idx = torch.minimum(wpt_idx + hit.long(), max_idx)

    new_prev_dist = active_dist.detach()
    return reward, new_wpt_idx, new_prev_dist, hit
