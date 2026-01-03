"""
Reward functions for discrete LfD trajectory following.
"""

from .reward_wp_v4_dis import waypoint_reward_v4_dis
from vttRL.tasks.robo_pp_lfd.rewards.reward_wp_v1 import waypoint_reward_v1
from vttRL.tasks.robo_pp_lfd.rewards.reward_wp_v2 import waypoint_reward_v2
from vttRL.tasks.robo_pp_lfd.rewards.reward_wp_v3 import waypoint_reward_v3
from vttRL.tasks.robo_pp_lfd.rewards.reward_wp_x import waypoint_reward_x
from vttRL.tasks.robo_pp_lfd.rewards.reward_wp_y import waypoint_reward_y

__all__ = [
    "waypoint_reward_v1",
    "waypoint_reward_v2",
    "waypoint_reward_v3",
    "waypoint_reward_v4_dis",
    "waypoint_reward_x",
    "waypoint_reward_y",
]
