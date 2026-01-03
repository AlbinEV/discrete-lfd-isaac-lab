"""
Minimal LfD polishing environment (torque-only).

Goals:
- Load a demonstration trajectory (waypoints in task space).
- Expose a torque action space (7 joints) and observations needed to follow the trajectory.
- Provide a reward that targets a constant contact force along Z (default -20 N).
- Manage simple phases: approach -> contact -> detach (based on force and last waypoint).
- Log desired vs executed trajectory (one JSON per episode for a chosen env).

No OSC controller, no fixed trajectory code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import field
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import DomeLightCfg, PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.lights import spawn_light
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from vttRL.tasks.robo_pp_lfd.Trajectory_Manager.LfD_trajectory import LfDTrajectoryManager, load_isaacsim_traj
from vttRL.tasks.robo_pp_lfd.cfg.config import F_TOUCH, F_LOST
from vttRL.tasks.robo_pp_lfd.cfg.robot_cfg import panda_cfg
from vttRL.tasks.robo_pp_lfd.cfg.scene_cfg import JustPushSceneCfg
from vttRL.tasks.robo_pp_lfd.rewards.wp_reward import waypoint_reward
from .rewards import (
    waypoint_reward_v1,
    waypoint_reward_v2,
    waypoint_reward_v3,
    waypoint_reward_v4_dis,
    waypoint_reward_x,
    waypoint_reward_y,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name, default)
    return val if val else default


def _env_floats(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        parts = [float(p.strip()) for p in raw.split(",") if p.strip()]
    except Exception:
        return default
    return parts if len(parts) == len(default) else default


# ---- spaces ----

SEQ_LEN = _env_int("LFD_SEQ_LEN", 64)


def _episode_length_s() -> float:
    max_steps = _env_int("LFD_MAX_EPISODE_STEPS", 0)
    if max_steps > 0:
        sim_dt = _env_float("LFD_SIM_DT", 1 / 120)
        decimation = _env_int("LFD_DECIMATION", 10)
        return max_steps * sim_dt * decimation
    return _env_float("LFD_EPISODE_LENGTH_S", 120.0)


def _make_action_space() -> spaces.Discrete:
    # Discrete 3^3 actions: (-1,0,+1) for x,y,z
    return spaces.Discrete(27)


def _make_observation_space() -> spaces.Box:
    # Per-step obs:
    # joint_pos(7) + joint_vel(7) + joint_acc(7) + joint_tau(7)
    # + ee_pos(3) + goal_pos(3) + fz(1) + phase(1) = 36
    one_step = 36
    return spaces.Box(-np.inf, np.inf, shape=(one_step * SEQ_LEN,), dtype=np.float32)


@configclass
class PolishLfDDiscreteEnvCfg(DirectRLEnvCfg):
    episode_length_s: float = _episode_length_s()
    decimation: int = _env_int("LFD_DECIMATION", 10)
    seq_length: int = SEQ_LEN

    action_space: spaces.Discrete = field(default_factory=_make_action_space)
    observation_space: spaces.Box = field(default_factory=_make_observation_space)
    state_space: spaces.Box = field(init=False)

    scene: JustPushSceneCfg = JustPushSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)

    sim: SimulationCfg = SimulationCfg(
        # Default to CPU to scale env count on low-VRAM machines; override with LFD_SIM_DEVICE=cuda:0
        device=_env_str("LFD_SIM_DEVICE", "cpu"),
        dt=_env_float("LFD_SIM_DT", 1 / 120),
        # Gravity disabled to simplify torque control while debugging trajectory tracking
        gravity=(0.0, 0.0, 0.0),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=32,
            max_velocity_iteration_count=1,
            gpu_max_rigid_contact_count=_env_int("LFD_GPU_MAX_RIGID_CONTACTS", 2**16),
        ),
        physics_material=RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0),
    )

    robot: ArticulationCfg = panda_cfg

    def __post_init__(self):
        self.state_space = self.observation_space


class PolishLfDDiscreteEnv(DirectRLEnv):
    """
    Discrete LfD env with task-space controller:
    - action: 27 discrete deltas (dx,dy,dz in {-1,0,+1} * step)
    - observation: stacked sequence of per-step features (see _make_observation_space)
    """

    def __init__(self, cfg: PolishLfDDiscreteEnvCfg, render_mode: str, **kwargs):
        self._trajectory_initialized = False
        # Randomization controls (define early to be available during init)
        self.wpt_noise_std = _env_float("LFD_WPT_NOISE_STD", 0.0)              # [m] Gaussian noise on waypoint positions
        self.init_joint_noise_std = _env_float("LFD_INIT_JOINT_NOISE_STD", 0.0)  # [rad] noise on initial joint pose
        self.init_joint_vel_std = _env_float("LFD_INIT_JOINT_VEL_STD", 0.0)      # [rad/s] noise on initial joint velocities
        self._env_origins = None
        # Torque limits will be initialized after scene is created
        self.torque_limit = None
        super().__init__(cfg, render_mode, **kwargs)
        # Align gymnasium observation_space with dict observations ("policy"/"critic")
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        if not self._trajectory_initialized:
            self._initialize_robot_components()

        self.seq_length = cfg.seq_length

        # Per-env state
        self.frame = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.wpt_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 0=approach,1=contact,2=detach
        # Waypoint coverage tracking
        self._wpt_max = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._wpt_hits = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Contact force tracking
        self._fz_ema = torch.zeros(self.num_envs, device=self.device)
        self._ema_alpha = _env_float("LFD_FZ_EMA_ALPHA", 0.1)
        self.fz_target = _env_float("LFD_FZ_TARGET", -20.0)
        self.fz_eps = _env_float("LFD_FZ_EPS", 2.0)

        # Action buffers
        self.action_dim = 7
        self.joint_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self.prev_joint_actions = torch.zeros_like(self.joint_actions)
        self.prev_joint_vel = torch.zeros_like(self.joint_actions)
        if self.torque_limit is None:
            self.torque_limit = torch.ones((self.action_dim,), device=self.device)
        default_joint_scale = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
        self._tau_joint_scale = torch.tensor(
            _env_floats("LFD_TAU_JOINT_SCALE", default_joint_scale),
            device=self.device,
            dtype=torch.float32,
        )
        if self._tau_joint_scale.numel() != self.action_dim:
            self._tau_joint_scale = torch.ones((self.action_dim,), device=self.device)
        print(f"[LFD TORQUE] Using joint torque scales: {self._tau_joint_scale.tolist()}")
        # Torque curriculum (global ramp)
        self._global_step = 0
        self._tau_scale_start = _env_float("LFD_TAU_SCALE_START", 0.2)
        self._tau_scale_end = _env_float("LFD_TAU_SCALE_END", 1.0)
        self._tau_scale_steps = max(1, _env_int("LFD_TAU_SCALE_STEPS", 10000))
        self._tau_scale_fixed = _env_float("LFD_TAU_SCALE_FIXED", -1.0)
        self._use_tau_scale_ramp = self._tau_scale_fixed < 0.0
        self._tau_scale = self._tau_scale_start
        self._tau_scale_jitter_std = _env_float("LFD_TAU_SCALE_JITTER_STD", 0.0)
        self._tau_scale_env = torch.ones((self.num_envs,), device=self.device)
        self._sample_tau_scale_env(torch.arange(self.num_envs, device=self.device))
        if not self._use_tau_scale_ramp:
            self._tau_scale = self._tau_scale_fixed
        self._tau_mode = _env_str("LFD_TAU_MODE", "direct").strip().lower()
        if self._tau_mode not in ("direct", "delta"):
            self._tau_mode = "direct"
        self._tau_delta_scale = _env_float("LFD_TAU_DELTA_SCALE", 1.0)
        default_delta_max = [0.1, 0.1, 0.1, 0.04, 0.04, 0.04, 0.04]
        self._tau_delta_max = torch.tensor(
            _env_floats("LFD_TAU_DELTA_MAX", default_delta_max),
            device=self.device,
            dtype=torch.float32,
        )
        print(f"[LFD TORQUE] Mode: {self._tau_mode}")
        if self._tau_mode == "delta":
            print(f"[LFD TORQUE] Delta scale: {self._tau_delta_scale}")
            print(f"[LFD TORQUE] Delta max: {self._tau_delta_max.tolist()}")
        # Discrete action settings (step in cm -> meters)
        self._discrete_step_cm = _env_float("LFD_DIS_STEP_CM", 0.1)
        self._discrete_step_m = self._discrete_step_cm * 0.01
        self._discrete_eps = _env_float("LFD_DIS_EPS", 0.002)
        self._discrete_hold = os.environ.get("LFD_DIS_HOLD", "1").lower() not in ("0", "false")
        self._discrete_action_mask = os.environ.get("LFD_DIS_ACTION_MASK", "1").lower() not in ("0", "false")
        self._discrete_mask_tol = _env_float("LFD_DIS_MASK_TOL", 0.0)
        self._ik_kp = _env_float("LFD_IK_KP", 80.0)
        self._ik_kd = _env_float("LFD_IK_KD", 2.0)
        self._last_delta = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_delta_raw = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_action_idx = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._last_applied_action_idx = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._last_cmd_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._cmd_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._cmd_pos_valid = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._last_hold = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._last_masked = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._cmd_clamp_margin = _env_float("LFD_CMD_CLAMP_MARGIN", 0.05)
        self._done_on_last = os.environ.get("LFD_DONE_ON_LAST", "0").lower() not in ("0", "false")
        self._traj_bounds_min = torch.zeros((self.num_envs, 3), device=self.device)
        self._traj_bounds_max = torch.zeros((self.num_envs, 3), device=self.device)
        self._tcp_offset = torch.tensor(
            _env_floats("LFD_TCP_OFFSET", [0.0, 0.0, 0.0]),
            device=self.device,
            dtype=torch.float32,
        )
        if self._tcp_offset.numel() != 3:
            self._tcp_offset = torch.zeros((3,), device=self.device)
        self._j1_offset_deg = _env_float("LFD_J1_OFFSET_DEG", 0.0)
        self._j1_offset_rad = float(np.deg2rad(self._j1_offset_deg))

        # Observation buffer (sequence)
        self.obs_dim = cfg.observation_space.shape[0] // cfg.seq_length
        self.obs_buffer = torch.zeros((self.num_envs, cfg.seq_length, self.obs_dim), device=self.device)
        
        # Reward config - Support versioned rewards (wp_v1, wp_v2, baseline, etc.)
        self.reward_mode = os.environ.get("LFD_REWARD_MODE", "wp_v4_dis")
        self._load_reward_function()
        self.prev_dist = torch.zeros(self.num_envs, device=self.device)

# Logging: single env only
        self._traj_log_enabled = os.environ.get("LFD_SAVE_TRAJ", "1").lower() not in ("0", "false")
        self._traj_log_dir = Path(os.environ.get("LFD_DEBUG_DIR", "logs/lfd_debug"))
        self._traj_log_env = int(os.environ.get("LFD_LOG_ENV", "0"))
        self._traj_log: list[dict] = []
        self._traj_episode_idx = 0

        # Components for logging / monitoring
        self._r_last = 0.0
        self._debug_check_goal = os.environ.get("LFD_DEBUG_CHECK_GOAL", "0").lower() not in ("0", "false")
        self._last_goal = None
        self._last_wpt_idx = -1
        self._last_wpt_idx_progress = -1

        # Debug prints (RT) controllabili da env
        self._debug_print = os.environ.get("LFD_DEBUG_PRINT", "0").lower() not in ("0", "false")
        self._debug_every = _env_int("LFD_DEBUG_EVERY", 50)
        self._debug_waypoint = os.environ.get("LFD_DEBUG_WAYPOINT", "0").lower() not in ("0", "false")
        self._debug_waypoint_every = _env_int("LFD_DEBUG_WAYPOINT_EVERY", 50)
        self._debug_tau = os.environ.get("LFD_DEBUG_TAU", "0").lower() not in ("0", "false")
        self._debug_tau_every = _env_int("LFD_DEBUG_TAU_EVERY", 50)
        self._tau_debug_counter = 0
        self._debug_joints = os.environ.get("LFD_DEBUG_JOINTS", "0").lower() not in ("0", "false")
        self._debug_joints_every = _env_int("LFD_DEBUG_JOINTS_EVERY", 200)
        self._debug_obs = os.environ.get("LFD_DEBUG_OBS", "0").lower() not in ("0", "false")
        self._debug_obs_every = _env_int("LFD_DEBUG_OBS_EVERY", 200)
        self._debug_x = os.environ.get("LFD_DEBUG_X", "0").lower() not in ("0", "false")
        self._debug_x_every = _env_int("LFD_DEBUG_X_EVERY", 50)
        self._debug_discrete = os.environ.get("LFD_DEBUG_DISCRETE", "0").lower() not in ("0", "false")
        self._debug_discrete_every = _env_int("LFD_DEBUG_DISCRETE_EVERY", 50)
        self._discrete_debug_counter = 0
        self._debug_discrete_post = os.environ.get("LFD_DEBUG_DISCRETE_POST", "0").lower() not in ("0", "false")
        self._debug_discrete_post_every = _env_int("LFD_DEBUG_DISCRETE_POST_EVERY", 50)
        self._prev_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_ee_valid = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._discrete_debug_post_counter = 0
        self._debug_reward = os.environ.get("LFD_DEBUG_REWARD", "0").lower() not in ("0", "false")
        self._debug_reward_every = _env_int("LFD_DEBUG_REWARD_EVERY", 50)
        self._debug_reward_all = os.environ.get("LFD_DEBUG_REWARD_ALL", "0").lower() not in ("0", "false")
        self._debug_reset = os.environ.get("LFD_DEBUG_RESET", "0").lower() not in ("0", "false")
        self._debug_progress = os.environ.get("LFD_DEBUG_PROGRESS", "0").lower() not in ("0", "false")
        self._debug_progress_every = _env_int("LFD_DEBUG_PROGRESS_EVERY", 100)
        self._debug_sanity = os.environ.get("LFD_DEBUG_SANITY", "0").lower() not in ("0", "false")
        self._debug_sanity_every = _env_int("LFD_DEBUG_SANITY_EVERY", 50)
        self._sanity_counter = 0
        # Stuck detection (terminate if too far for too long)
        self._stuck_dist = _env_float("LFD_STUCK_DIST", 0.05)   # 5 cm default
        # default: allow as many steps as an episode to avoid early truncation; override via LFD_STUCK_STEPS
        default_stuck_steps = int(getattr(self, "max_episode_length", 0) or 0) or 200
        self._stuck_steps = _env_int("LFD_STUCK_STEPS", default_stuck_steps)

        # Waypoint timeout: advance even se non siamo vicini
        self._wpt_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._wpt_timeout = _env_int("LFD_WP_TIMEOUT", 50)
        self._wpt_eps = _env_float("LFD_WP_EPS", 0.01)  # 1 cm sfera di arrivo
        # Stuck counter
        self._stuck_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Optional phase usage (contact not needed for pure trajectory following)
        self._use_phase = os.environ.get("LFD_USE_PHASE", "0").lower() not in ("0", "false")
        disable_stuck_env = os.environ.get("LFD_DISABLE_STUCK")
        if disable_stuck_env is None:
            # Default: stuck detection attiva (solo se l'utente non la disabilita esplicitamente).
            self._disable_stuck = False
        else:
            self._disable_stuck = disable_stuck_env.lower() not in ("0", "false")

    # --- logging helpers ---
    def _load_reward_function(self):
        """
        Load reward function based on LFD_REWARD_MODE environment variable.
        Supports versioned rewards: wp_v1, wp_v2, wp_v3, wp_v4_dis, wp_x, or legacy wp/baseline modes.
        """
        mode = self.reward_mode.lower()
        
        # Map reward modes to functions
        if mode == "wp_v1":
            self.reward_fn = waypoint_reward_v1
            print(f"[LFD REWARD] Using waypoint_reward_v1 (baseline without time penalty)")
        elif mode == "wp_v2":
            self.reward_fn = waypoint_reward_v2
            print(f"[LFD REWARD] Using waypoint_reward_v2 (with time penalty)")
        elif mode == "wp_v3":
            self.reward_fn = waypoint_reward_v3
            print("[LFD REWARD] Using waypoint_reward_v3 (sequential axis tracking)")
        elif mode == "wp_v4_dis":
            self.reward_fn = waypoint_reward_v4_dis
            print("[LFD REWARD] Using waypoint_reward_v4_dis (discrete, sequential axis tracking)")
        elif mode in ("wp_x", "wp_x_only"):
            self.reward_fn = waypoint_reward_x
            print("[LFD REWARD] Using waypoint_reward_x (X axis only)")
        elif mode in ("wp_y", "wp_y_only"):
            self.reward_fn = waypoint_reward_y
            print("[LFD REWARD] Using waypoint_reward_y (Y axis only)")
        elif mode == "wp":
            # Legacy mode: use original wp_reward.py (same as v1)
            self.reward_fn = waypoint_reward
            print(f"[LFD REWARD] Using waypoint_reward (legacy, equivalent to v1)")
        else:
            # baseline or other modes don't use waypoint_reward
            self.reward_fn = None
            print(f"[LFD REWARD] Using baseline reward mode: {mode}")

    def _update_reward_log(self, env_id: int, comp: dict[str, float]) -> None:
        if not self._traj_log:
            return
        if env_id != self._traj_log_env:
            return
        entry = self._traj_log[-1]
        reward_total = comp.get("reward_total")
        if reward_total is not None:
            entry["reward_total"] = reward_total
        reward_components = {k: v for k, v in comp.items() if k != "reward_total"}
        if reward_components:
            entry["reward_components"] = reward_components

    def _find_vis_script(self) -> Path | None:
        here = Path(__file__).resolve()
        for parent in [here.parent] + list(here.parents):
            candidate = parent / "scripts" / "visualize_trajectory_3d.py"
            if candidate.is_file():
                return candidate
        return None

    def _write_traj_html(self, json_path: Path) -> None:
        html_root = os.environ.get("LFD_HTML_DIR", "")
        html_dir = Path(html_root) if html_root else json_path.parent
        html_dir.mkdir(parents=True, exist_ok=True)
        script = self._find_vis_script()
        if script is None:
            print("[LFD WARN] visualize_trajectory_3d.py not found; skip HTML generation.")
            return
        exec_html = html_dir / f"{json_path.stem}.exec_full.html"
        target_html = html_dir / f"{json_path.stem}.target_full.html"
        base_cmd = [sys.executable, str(script), "--file", str(json_path)]
        for extra_args, out_path in (([], exec_html), (["--no_exec"], target_html)):
            cmd = base_cmd + ["--save", str(out_path)] + extra_args
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                stderr = result.stderr.strip() if result.stderr else "unknown error"
                print(f"[LFD WARN] HTML generation failed ({out_path.name}): {stderr}")

    def _flush_traj_log(self, force: bool = False):
        """Scrive il log corrente in JSON compatibile con visualize_trajectory_3d.py."""
        if not self._traj_log_enabled:
            return
        # Se non ci sono dati, non scrivere nulla (anche se force=True)
        if not self._traj_log:
            return
        episode_steps = len(self._traj_log)
        traj_len = int(getattr(self.traj_mgr, "T", 0)) if hasattr(self, "traj_mgr") else 0
        # Salta log che non hanno fatto progressi sui waypoint (tipico di flush duplicati/reset immediati).
        wpt_max_val = int(self._wpt_max[self._traj_log_env].item()) if traj_len > 1 else 0
        if traj_len > 1 and (episode_steps <= 1 or wpt_max_val <= 0):
            self._traj_log = []
            return
        self._traj_log_dir.mkdir(parents=True, exist_ok=True)
        self._traj_episode_idx += 1
        out_path = self._traj_log_dir / f"lfd_exec_env{self._traj_log_env:03d}_{self._traj_episode_idx:05d}.json"
        dt = float(getattr(self, "physics_dt", self.cfg.sim.dt))
        isaac_data = []
        for r in self._traj_log:
            reward_total = r.get("reward_total")
            reward_components = r.get("reward_components")
            data = {
                "target_position": [r["x_des"], r["y_des"], r["z_des"]],
                "ee_position": [r["x_act"], r["y_act"], r["z_act"]],
                "cmd_position": [r.get("x_cmd"), r.get("y_cmd"), r.get("z_cmd")],
                "delta_cmd": [r.get("dx_cmd"), r.get("dy_cmd"), r.get("dz_cmd")],
                "raw_delta_cmd": [r.get("raw_dx_cmd"), r.get("raw_dy_cmd"), r.get("raw_dz_cmd")],
                "action_idx": r.get("action_idx"),
                "applied_action_idx": r.get("applied_action_idx"),
                "hold": r.get("hold"),
                "masked": r.get("masked"),
                "joint_positions": r["joint_positions"],
                "applied_torques": [r[f"tau_{i}"] for i in range(7)],
                "phase": r["phase"],
                "waypoint_index": r["wpt_idx"],
                "force_z": r["fz"],
            }
            if reward_total is not None:
                data["reward_total"] = reward_total
            if reward_components:
                data["reward_components"] = reward_components
            isaac_data.append(
                {
                    "current_time": r.get("current_time", r["frame"] * dt),
                    "current_time_step": r["frame"],
                    "data": data,
                }
            )
        metadata = {
            "test_tag": os.environ.get("LFD_TEST_TAG", ""),
            "test_cfg": os.environ.get("LFD_TEST_CFG", ""),
            "reward_mode": self.reward_mode,
            "use_phase": int(getattr(self, "_use_phase", False)),
            "disable_stuck": int(getattr(self, "_disable_stuck", False)),
            "traj_path": str(getattr(self, "_traj_path", "")),
            "traj_z_offset": float(getattr(self, "_traj_z_offset", 0.0)),
            "traj_dedup": bool(getattr(self, "_traj_dedup", True)),
            "traj_len": int(getattr(self.traj_mgr, "T", 0)) if hasattr(self, "traj_mgr") else 0,
            "discrete_step_cm": float(getattr(self, "_discrete_step_cm", 0.0)),
            "discrete_eps": float(getattr(self, "_discrete_eps", 0.0)),
            "discrete_hold": int(getattr(self, "_discrete_hold", False)),
            "action_mask": int(getattr(self, "_discrete_action_mask", False)),
            "mask_tol": float(getattr(self, "_discrete_mask_tol", 0.0)),
            "cmd_clamp_margin": float(getattr(self, "_cmd_clamp_margin", 0.0)),
            "done_on_last": int(getattr(self, "_done_on_last", False)),
            "tcp_offset": [float(v) for v in getattr(self, "_tcp_offset", torch.zeros(3, device=self.device)).tolist()],
            "j1_offset_deg": float(getattr(self, "_j1_offset_deg", 0.0)),
            "j1_offset_rad": float(getattr(self, "_j1_offset_rad", 0.0)),
            "ik_kp": float(getattr(self, "_ik_kp", 0.0)),
            "ik_kd": float(getattr(self, "_ik_kd", 0.0)),
            "w_cmd_err": _env_float("LFD_W_CMD_ERR", 0.0),
            "w_cmd_hit": _env_float("LFD_W_CMD_HIT", 0.0),
            "w_wpt_adv": _env_float("LFD_W_WPT_ADV", 0.0),
            "w_wpt_cover": _env_float("LFD_W_WPT_COVER", 0.0),
            "cmd_eps": _env_float("LFD_CMD_EPS", float(getattr(self, "_discrete_eps", 0.0))),
            "w_done_success": _env_float("LFD_W_DONE_SUCCESS", 100_000_000.0),
            "w_done_early": _env_float("LFD_W_DONE_EARLY", 0.0),
            "w_wp": _env_float("LFD_W_WP", 1.0),
            "w_prog": _env_float("LFD_W_PROG", 2.0),
            "w_hit": _env_float("LFD_W_HIT", 5.0),
            "w_time": _env_float("LFD_W_TIME", -0.1),
            "w_dtau": _env_float("LFD_W_DTAU", 0.0),
            "w_stuck": _env_float("LFD_W_STUCK", 0.0),
        }
        # Episode stats for termination analysis
        metadata["episode_steps"] = int(episode_steps)
        metadata["max_episode_length"] = int(getattr(self, "max_episode_length", 0))
        if metadata["max_episode_length"] > 0:
            metadata["terminated_early"] = int(metadata["episode_steps"] < metadata["max_episode_length"])
        # Coverage summary for the logged env
        if traj_len > 1:
            metadata["wpt_max"] = wpt_max_val
            metadata["wpt_hits"] = int(self._wpt_hits[self._traj_log_env].item())
            metadata["wpt_coverage"] = float(wpt_max_val / (traj_len - 1))
        payload = {"metadata": metadata, "Isaac Sim Data": isaac_data}
        out_path.write_text(json.dumps(payload, indent=2))
        self._write_traj_html(out_path)
        self._traj_log = []

    # --- scene ---
    def _setup_scene(self):
        # Try default ground asset; fallback to procedural plane if missing
        try:
            spawn_ground_plane("/World/ground", GroundPlaneCfg(), translation=(0, 0, 0))
        except Exception as exc:
            print(f"[WARN] Failed to load default ground plane, using procedural plane: {exc}")
            from isaaclab.assets import RigidObjectCfg
            from isaaclab.utils.geometry import YawPitchRoll
            import isaaclab.sim as sim_utils
            ground_cfg = RigidObjectCfg(
                prim_path="/World/ground",
                spawn=sim_utils.CuboidCfg(
                    size=(5.0, 5.0, 0.01),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, -0.005),
                    rot=(0.0, 0.0, 0.0, 1.0),
                ),
            )
            ground_cfg.spawn()
        self.robot = Articulation(self.cfg.robot)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        spawn_light("/World/Light", DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)))

    def _initialize_robot_components(self):
        if self._trajectory_initialized:
            return

        # Contact sensor from scene (provided by JustPushSceneCfg)
        self.carpet_sensor = self.scene["carpet_contact"]

        # Joint ids
        joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        self.joint_ids, _ = self.robot.find_joints(joint_names)

        # EE body index
        bodies, _ = self.robot.find_bodies(["EE"])
        self.ee_body_idx = bodies[0]
        self.ee_jac_idx = bodies[0] - 1

        diff_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            ik_method="dls",
            ik_params={
                "lambda_val": _env_float("LFD_IK_LAMBDA", 0.1),
                "pos_weight": _env_float("LFD_IK_POS_W", 1.0),
                "rot_weight": _env_float("LFD_IK_ROT_W", 0.0),
            },
        )
        self.diffik = DifferentialIKController(diff_cfg, num_envs=self.num_envs, device=self.device)

        # Torque limits (best-effort)
        if hasattr(self.robot, "actuators") and self.robot.actuators:
            shoulder_lim = self.robot.actuators["panda_shoulder"].cfg.effort_limit if "panda_shoulder" in self.robot.actuators else 0.0
            forearm_lim = self.robot.actuators["panda_forearm"].cfg.effort_limit if "panda_forearm" in self.robot.actuators else 0.0
            limits = torch.tensor([shoulder_lim] * 4 + [forearm_lim] * 3, dtype=torch.float32, device=self.device)
        else:
            limits = self.torque_limit
            if limits is None:
                limits = torch.ones((7,), dtype=torch.float32, device=self.device)
        # Fallback: if limits are too small/unset, use nominal Panda values
        use_fallback = torch.all(limits <= 1.1)
        if use_fallback:
            limits = torch.tensor([87.0] * 4 + [12.0] * 3, dtype=torch.float32, device=self.device)

        # Optional overrides
        override_raw = _env_str("LFD_TAU_LIMIT_OVERRIDE", "")
        disable_limits = _env_int("LFD_DISABLE_TAU_LIMITS", 0) != 0
        override_ok = False
        if disable_limits:
            limits = torch.ones((7,), dtype=torch.float32, device=self.device)
        elif override_raw:
            parts = [p.strip() for p in override_raw.split(",") if p.strip()]
            try:
                if len(parts) == 1:
                    val = float(parts[0])
                    limits = torch.full((7,), val, dtype=torch.float32, device=self.device)
                    override_ok = True
                elif len(parts) == 7:
                    vals = [float(p) for p in parts]
                    limits = torch.tensor(vals, dtype=torch.float32, device=self.device)
                    override_ok = True
                else:
                    raise ValueError("Expected 1 or 7 values.")
            except Exception:
                override_ok = False

        limit_scale = _env_float("LFD_TAU_LIMIT_SCALE", 1.0)
        if limit_scale != 1.0:
            limits = limits * limit_scale

        if disable_limits:
            print(
                f"[LFD TORQUE] Torque limits disabled; using unit limits (scaled by {limit_scale:g}): {limits.tolist()}"
            )
        elif override_ok:
            print(f"[LFD TORQUE] Using torque limit override: {limits.tolist()}")
        elif use_fallback:
            print(f"[LFD TORQUE] Using fallback torque limits (Panda): {limits.tolist()}")
        else:
            print(f"[LFD TORQUE] Using actuator torque limits: {limits.tolist()}")
        self.torque_limit = limits

        # Trajectory (local JSON, override via env LFD_TRAJ_PATH)
        traj_path_env = os.environ.get("LFD_TRAJ_PATH", "").strip()
        if traj_path_env:
            traj_path = Path(traj_path_env).expanduser()
            if not traj_path.is_file():
                raise FileNotFoundError(f"LFD_TRAJ_PATH does not exist: {traj_path}")
            print(f"[LFD TRAJ] Using custom trajectory from LFD_TRAJ_PATH={traj_path}")
        else:
            traj_path = Path(__file__).resolve().parent / "data_traj_20250506.json"
            if not traj_path.is_file():
                traj_path = Path(__file__).resolve().parents[1] / "robo_pp_lfd" / "data_traj_20250506.json"
            print(f"[LFD TRAJ] Using default trajectory: {traj_path.name}")
        # Default dedup disattivato per preservare tutti i waypoint; abilita con LFD_DEDUP=1.
        dedup = os.environ.get("LFD_DEDUP", "0").lower() not in ("0", "false")
        z_offset = _env_float("LFD_TRAJ_Z_OFFSET", 0.0)
        self._traj_path = traj_path
        self._traj_dedup = dedup
        self._traj_z_offset = z_offset
        ee_goals = load_isaacsim_traj(traj_path, z_offset=z_offset, dedup=dedup)
        self.traj_mgr = LfDTrajectoryManager(base_goals=ee_goals, num_envs=self.num_envs, device=self.device)
        env_ids_t = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self.traj_mgr.sample(env_ids_t)
        self._apply_traj_noise(env_ids_t)
        self._update_traj_bounds(env_ids_t)
        if os.environ.get("LFD_DEBUG_TRAJ", "0").lower() not in ("0", "false"):
            step_cm = _env_float("LFD_DIS_STEP_CM", 0.1)
            eps = _env_float("LFD_DIS_EPS", 0.002)
            hold = os.environ.get("LFD_DIS_HOLD", "1").lower() not in ("0", "false")
            clamp_margin = _env_float("LFD_CMD_CLAMP_MARGIN", 0.05)
            print(
                f"[LFD TRAJ DEBUG] xy_random_range={getattr(self.traj_mgr, 'xy_randomization_range', None)} "
                f"z_offset={z_offset:.4f} step_cm={step_cm:.2f} "
                f"eps={eps:.4f} hold={int(hold)} clamp_margin={clamp_margin:.3f}",
                flush=True,
            )

        # Optional debug on trajectory stats
        if os.environ.get("LFD_DEBUG_TRAJ", "0").lower() not in ("0", "false"):
            p = self.traj_mgr.p_traj  # (T,3)
            print(
                f"[LFD TRAJ DEBUG] Loaded {self.traj_mgr.T} waypoints | "
                f"X range: {p[:,0].min().item():.4f}/{p[:,0].max().item():.4f} | "
                f"Y range: {p[:,1].min().item():.4f}/{p[:,1].max().item():.4f} | "
                f"Z range: {p[:,2].min().item():.4f}/{p[:,2].max().item():.4f}"
            )
            try:
                p0 = self.traj_mgr.p_traj_env[0, :5].detach().cpu().tolist()
                print(f"[LFD TRAJ DEBUG] First 5 waypoints (env0): {p0}")
            except Exception:
                pass

        self._trajectory_initialized = True

    # --- core loop ---
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Handle None on first reset
        if actions is None:
            actions = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        # Update global torque scale (curriculum) if enabled
        if self._use_tau_scale_ramp:
            self._global_step += 1
            progress = min(1.0, float(self._global_step) / float(self._tau_scale_steps))
            self._tau_scale = self._tau_scale_start + (self._tau_scale_end - self._tau_scale_start) * progress

        # Stash latest discrete action buffer and apply controller
        self.action_buf = actions.to(self.device).view(-1)
        self._apply_action()

    def _apply_traj_noise(self, env_ids):
        """Add Gaussian noise to waypoints for the given env ids."""
        std = getattr(self, "wpt_noise_std", 0.0)
        if std <= 0:
            return
        try:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)
            # Add noise on top of the already sampled per-env trajectory
            noise = torch.randn((len(env_ids_t), self.traj_mgr.T, 3), device=self.device) * std
            self.traj_mgr.p_traj_env[env_ids_t] = self.traj_mgr.p_traj_env[env_ids_t] + noise
        except Exception as exc:
            if getattr(self, "_debug_reset", False):
                print(f"[LFD NOISE] Failed to apply waypoint noise: {exc}", flush=True)

    def _update_traj_bounds(self, env_ids: torch.Tensor) -> None:
        """Compute per-env trajectory bounds for command clamping."""
        try:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)
            p = self.traj_mgr.p_traj_env[env_ids_t]  # (N, T, 3)
            self._traj_bounds_min[env_ids_t] = p.min(dim=1).values
            self._traj_bounds_max[env_ids_t] = p.max(dim=1).values
        except Exception as exc:
            if getattr(self, "_debug_reset", False):
                print(f"[LFD BOUNDS] Failed to update bounds: {exc}", flush=True)

    def _sample_tau_scale_env(self, env_ids: torch.Tensor) -> None:
        """Sample per-env torque scale jitter (multiplicative)."""
        if self._tau_scale_jitter_std <= 0:
            self._tau_scale_env[env_ids] = 1.0
            return
        jitter = torch.randn((len(env_ids),), device=self.device) * self._tau_scale_jitter_std
        scale = torch.clamp(1.0 + jitter, min=0.05)
        self._tau_scale_env[env_ids] = scale

    def _post_physics_step(self):
        # Intentionally empty; frame is advanced in _get_observations to match IsaacLab 2.0.1 call order.
        pass

    def _get_env_origins(self) -> torch.Tensor:
        if self._env_origins is None:
            origins = getattr(self.scene, "env_origins", None)
            if origins is None:
                self._env_origins = torch.zeros((self.num_envs, 3), device=self.device)
            else:
                if torch.is_tensor(origins):
                    origins = origins.to(self.device)
                else:
                    origins = torch.as_tensor(origins, dtype=torch.float32, device=self.device)
                self._env_origins = origins
        return self._env_origins

    def _discrete_to_delta(self, actions: torch.Tensor) -> torch.Tensor:
        """Map discrete action indices to XYZ delta in meters."""
        act = actions.to(self.device).long().view(-1)
        act = torch.clamp(act, 0, 26)
        vals = torch.tensor([-1.0, 0.0, 1.0], device=self.device)
        ix = act // 9
        iy = (act // 3) % 3
        iz = act % 3
        delta = torch.stack((vals[ix], vals[iy], vals[iz]), dim=1) * self._discrete_step_m
        return delta

    def _apply_action(self) -> None:
        # Discrete XYZ delta -> IK -> joint torque PD
        env_origins = self._get_env_origins()
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_idx]
        ee_pos = ee_pos_w - env_origins
        ee_quat = self.robot.data.body_quat_w[:, self.ee_body_idx]
        if not torch.all(self._cmd_pos_valid):
            invalid = ~self._cmd_pos_valid
            self._cmd_pos[invalid] = ee_pos[invalid]
            self._cmd_pos_valid[invalid] = True
        delta = self._discrete_to_delta(self.action_buf)
        self._last_delta_raw = delta
        self._last_action_idx = self.action_buf.to(self.device).long().view(-1)
        applied_delta = delta.clone()
        applied_action_idx = self._last_action_idx.clone()
        if self._discrete_hold:
            cmd_err = torch.linalg.vector_norm(self._cmd_pos - ee_pos, dim=1)
            allow = cmd_err <= self._discrete_eps
            applied_delta[~allow] = 0.0
            applied_action_idx[~allow] = -1
            self._last_hold = ~allow
        else:
            self._last_hold = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._last_masked = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if self._discrete_action_mask:
            env_ids = torch.arange(self.num_envs, device=self.device)
            goal_pos, _ = self.traj_mgr.get_targets(env_ids, self.wpt_idx)
            curr_cmd = self._cmd_pos
            cand_cmd = curr_cmd + applied_delta
            curr_dist = torch.linalg.vector_norm(goal_pos - curr_cmd, dim=1)
            cand_dist = torch.linalg.vector_norm(goal_pos - cand_cmd, dim=1)
            worse = cand_dist > (curr_dist + self._discrete_mask_tol)
            out_of_bounds = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            if self._cmd_clamp_margin >= 0:
                min_bounds = self._traj_bounds_min - self._cmd_clamp_margin
                max_bounds = self._traj_bounds_max + self._cmd_clamp_margin
                out_of_bounds = torch.any((cand_cmd < min_bounds) | (cand_cmd > max_bounds), dim=1)
            masked = worse | out_of_bounds
            if torch.any(masked):
                applied_delta[masked] = 0.0
                applied_action_idx[masked] = -1
            self._last_masked = masked
        self._cmd_pos = self._cmd_pos + applied_delta
        # Clamp command within trajectory bounds (+ margin)
        if self._cmd_clamp_margin >= 0:
            min_bounds = self._traj_bounds_min - self._cmd_clamp_margin
            max_bounds = self._traj_bounds_max + self._cmd_clamp_margin
            self._cmd_pos = torch.max(torch.min(self._cmd_pos, max_bounds), min_bounds)
        self._last_delta = applied_delta
        self._last_applied_action_idx = applied_action_idx
        target_pos = self._cmd_pos
        self._last_cmd_pos = self._cmd_pos
        target_pos_w = target_pos + env_origins
        cmd = torch.cat((target_pos_w, ee_quat), dim=1)

        self.diffik.set_command(cmd, ee_pos=ee_pos_w, ee_quat=ee_quat)
        jac_w = self.robot.root_physx_view.get_jacobians()[:, self.ee_jac_idx, :, self.joint_ids]
        q_des = self.diffik.compute(ee_pos_w, ee_quat, jac_w, self.robot.data.joint_pos[:, self.joint_ids])

        q = self.robot.data.joint_pos[:, self.joint_ids]
        qd = self.robot.data.joint_vel[:, self.joint_ids]
        delta_q = q_des - q
        q_trg = q_des
        tau = self._ik_kp * (q_trg - q) - self._ik_kd * qd
        tau = tau * (self._tau_scale * self._tau_scale_env.unsqueeze(1))
        tau_limit = self.torque_limit * self._tau_joint_scale
        tau = torch.clamp(tau, -tau_limit, tau_limit)
        self.joint_actions = tau
        self.robot.set_joint_effort_target(tau, joint_ids=self.joint_ids)
        if self._debug_discrete and self.num_envs > 0:
            self._discrete_debug_counter += 1
            if self._discrete_debug_counter % max(self._debug_discrete_every, 1) == 0:
                e = 0
                dq_norm = float(torch.linalg.vector_norm(delta_q[e]).item())
                tau_norm = float(torch.linalg.vector_norm(tau[e]).item())
                print(
                    f"[LFD DIS] step={int(self.frame[e].item())} act={int(self._last_action_idx[e].item())} "
                    f"delta_raw=({delta[e,0]:+.4f},{delta[e,1]:+.4f},{delta[e,2]:+.4f}) "
                    f"delta_applied=({applied_delta[e,0]:+.4f},{applied_delta[e,1]:+.4f},{applied_delta[e,2]:+.4f}) "
                    f"ee=({ee_pos[e,0]:.3f},{ee_pos[e,1]:.3f},{ee_pos[e,2]:.3f}) "
                    f"cmd=({target_pos[e,0]:.3f},{target_pos[e,1]:.3f},{target_pos[e,2]:.3f}) "
                    f"hold={int(self._last_hold[e].item())} mask={int(self._last_masked[e].item())} "
                    f"dq_norm={dq_norm:.4f} tau_norm={tau_norm:.4f}",
                    flush=True,
                )
        if self._debug_tau and self.num_envs > 0:
            self._tau_debug_counter += 1
            if self._tau_debug_counter % max(self._debug_tau_every, 1) == 0:
                e = 0
                env_origins = self._get_env_origins()
                ee_pos_w = self.robot.data.body_pos_w[e, self.ee_body_idx]
                ee_pos = ee_pos_w - env_origins[e]
                env_ids = torch.tensor([e], device=self.device, dtype=torch.long)
                goal_pos, _ = self.traj_mgr.get_targets(env_ids, self.wpt_idx[env_ids])
                tau_norm = tau[e] / tau_limit
                if self._tau_mode == "delta":
                    dtau = (tau[e] - self.prev_joint_actions[e]).tolist()
                    dtau_str = f" dtau={dtau} dscale={self._tau_delta_scale:.3f}"
                else:
                    dtau_str = ""
                print(
                    f"[LFD TAU] step={self._tau_debug_counter} scale={self._tau_scale:.3f} "
                    f"env_scale={self._tau_scale_env[e].item():.3f} "
                    f"ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}) "
                    f"goal=({goal_pos[0,0]:.3f},{goal_pos[0,1]:.3f},{goal_pos[0,2]:.3f}) "
                    f"tau={tau[e].tolist()}{dtau_str} norm={tau_norm.tolist()} limit={tau_limit.tolist()}",
                    flush=True,
                )

    def _compute_phase(self, fz: torch.Tensor):
        if not self._use_phase:
            self.phase[:] = 0
            return
        # approach -> contact on touch; contact -> detach at last waypoint or force lost.
        at_last = self.wpt_idx >= (self.traj_mgr.T - 1)
        to_contact = (self.phase == 0) & (fz < F_TOUCH)
        self.phase[to_contact] = 1
        to_detach = (self.phase == 1) & (at_last | (fz > F_LOST))
        self.phase[to_detach] = 2

    def _advance_waypoints(self, ee_pos: torch.Tensor, goal_pos: torch.Tensor):
        # Advance only when the active condition is met.
        mode = self.reward_mode.lower()
        if mode in ("wp_x", "wp_x_only"):
            close = torch.abs(goal_pos[:, 0] - ee_pos[:, 0]) < self._wpt_eps
        elif mode in ("wp_y", "wp_y_only"):
            close = torch.abs(goal_pos[:, 1] - ee_pos[:, 1]) < self._wpt_eps
        else:
            # Default: sphere around waypoint
            delta = goal_pos - ee_pos
            dist = torch.linalg.vector_norm(delta, dim=1)
            close = dist < self._wpt_eps
        max_idx = self.traj_mgr.T - 1

        # Increment waypoint on proximity
        advanced = close.long()
        self.wpt_idx = torch.clamp(self.wpt_idx + advanced, 0, max_idx)

        # Update per-env dwell counters (no forced advance)
        stayed = (advanced == 0)
        self._wpt_step_count[stayed] += 1
        self._wpt_step_count[~stayed] = 0

        # Optional timeout: force-advance if stuck too long on a waypoint
        if self._wpt_timeout > 0:
            timed_out = self._wpt_step_count >= self._wpt_timeout
            if torch.any(timed_out):
                self.wpt_idx = torch.clamp(self.wpt_idx + timed_out.long(), 0, max_idx)
                self._wpt_step_count[timed_out] = 0

    def _get_observations(self):
        # Increment frame counter here to ensure it advances even if _post_physics_step is not invoked
        self.frame += 1
        env_origins = self._get_env_origins()
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_idx]
        ee_pos = ee_pos_w - env_origins
        goal_pos, _ = self.traj_mgr.get_targets(torch.arange(self.num_envs, device=self.device), self.wpt_idx)
        # Update stuck counter: if far from goal, increment; else reset
        mode = self.reward_mode.lower()
        if mode in ("wp_x", "wp_x_only"):
            dist_to_goal = torch.abs(goal_pos[:, 0] - ee_pos[:, 0])
        elif mode in ("wp_y", "wp_y_only"):
            dist_to_goal = torch.abs(goal_pos[:, 1] - ee_pos[:, 1])
        else:
            dist_to_goal = torch.linalg.vector_norm(goal_pos - ee_pos, dim=1)
        far = dist_to_goal > self._stuck_dist
        self._stuck_count[far] += 1
        self._stuck_count[~far] = 0

        # Force
        self.carpet_sensor.update(self.physics_dt * self.cfg.decimation)
        f_ext_w = self.carpet_sensor.data.net_forces_w[:, 0]
        fz_raw = f_ext_w[:, 2]
        self._fz_ema = (1.0 - self._ema_alpha) * self._fz_ema + self._ema_alpha * fz_raw

        self._compute_phase(self._fz_ema)
        if self.reward_fn is None:
            self._advance_waypoints(ee_pos, goal_pos)
            # Recompute goal after potential waypoint advance
            goal_pos, _ = self.traj_mgr.get_targets(torch.arange(self.num_envs, device=self.device), self.wpt_idx)

        # Build per-step obs
        q = self.robot.data.joint_pos[:, self.joint_ids]
        qd = self.robot.data.joint_vel[:, self.joint_ids]
        step_dt = float(getattr(self, "physics_dt", self.cfg.sim.dt)) * float(self.cfg.decimation)
        qdd = (qd - self.prev_joint_vel) / max(step_dt, 1e-6)
        self.prev_joint_vel = qd.clone()
        phase_norm = (self.phase.float() / 2.0).unsqueeze(1)
        obs_step = torch.cat(
            [q, qd, qdd, self.joint_actions, ee_pos, goal_pos, self._fz_ema.unsqueeze(1), phase_norm], dim=1
        )

        # Roll sequence buffer
        self.obs_buffer = torch.roll(self.obs_buffer, shifts=-1, dims=1)
        self.obs_buffer[:, -1, :] = obs_step
        obs = self.obs_buffer.reshape(self.num_envs, -1)

        # Optional sanity check: if waypoint index changes but goal stays identical, flag it
        if self._debug_check_goal and self._traj_log_env < self.num_envs:
            e = self._traj_log_env
            curr_goal = tuple(float(v) for v in goal_pos[e].detach().cpu().tolist())
            curr_idx = int(self.wpt_idx[e].item())
            if self._last_goal is not None and curr_idx != self._last_wpt_idx and curr_goal == self._last_goal:
                print(f"[LFD WARN] wpt_idx changed {self._last_wpt_idx}->{curr_idx} but goal unchanged {curr_goal}", flush=True)
            self._last_goal = curr_goal
            self._last_wpt_idx = curr_idx

        # Optional per-step logging (single env)
        if self._traj_log_enabled and self._traj_log_env < self.num_envs:
            e = self._traj_log_env
            dt = float(getattr(self, "physics_dt", self.cfg.sim.dt))
            self._traj_log.append(
                {
                    "frame": int(self.frame[e].item()),
                    "current_time": float(self.frame[e].item()) * dt,
                    "wpt_idx": int(self.wpt_idx[e].item()),
                    "phase": int(self.phase[e].item()),
                    "x_des": float(goal_pos[e, 0].item()),
                    "y_des": float(goal_pos[e, 1].item()),
                    "z_des": float(goal_pos[e, 2].item()),
                    "x_act": float(ee_pos[e, 0].item()),
                    "y_act": float(ee_pos[e, 1].item()),
                    "z_act": float(ee_pos[e, 2].item()),
                    "x_cmd": float(self._last_cmd_pos[e, 0].item()),
                    "y_cmd": float(self._last_cmd_pos[e, 1].item()),
                    "z_cmd": float(self._last_cmd_pos[e, 2].item()),
                    "action_idx": int(self._last_action_idx[e].item()),
                    "applied_action_idx": int(self._last_applied_action_idx[e].item()),
                    "dx_cmd": float(self._last_delta[e, 0].item()),
                    "dy_cmd": float(self._last_delta[e, 1].item()),
                    "dz_cmd": float(self._last_delta[e, 2].item()),
                    "raw_dx_cmd": float(self._last_delta_raw[e, 0].item()),
                    "raw_dy_cmd": float(self._last_delta_raw[e, 1].item()),
                    "raw_dz_cmd": float(self._last_delta_raw[e, 2].item()),
                    "hold": int(self._last_hold[e].item()),
                    "masked": int(self._last_masked[e].item()),
                    "joint_positions": [float(v) for v in q[e].tolist()],
                    "fz": float(self._fz_ema[e].item()),
                    "tau_0": float(self.joint_actions[e, 0].item()),
                    "tau_1": float(self.joint_actions[e, 1].item()),
                    "tau_2": float(self.joint_actions[e, 2].item()),
                    "tau_3": float(self.joint_actions[e, 3].item()),
                    "tau_4": float(self.joint_actions[e, 4].item()),
                    "tau_5": float(self.joint_actions[e, 5].item()),
                    "tau_6": float(self.joint_actions[e, 6].item()),
                }
            )

        # Optional debug printing
        if self._debug_print and (self.frame[0].item() % max(self._debug_every, 1) == 0):
            print(
                f"[LFD DEBUG] frame={int(self.frame[0].item())} "
                f"time={float(self.frame[0].item()) * dt:.4f} "
                f"wpt_idx={int(self.wpt_idx[0].item())} "
                f"phase={int(self.phase[0].item())} "
                f"goal=({goal_pos[0,0]:.3f},{goal_pos[0,1]:.3f},{goal_pos[0,2]:.3f}) "
                f"(wpt={int(self.wpt_idx[0].item())}) "
                f"ee=({ee_pos[0,0]:.3f},{ee_pos[0,1]:.3f},{ee_pos[0,2]:.3f}) "
                f"fz={float(self._fz_ema[0].item()):.3f}",
                flush=True,
            )
        if self._debug_waypoint and (self.frame[0].item() % max(self._debug_waypoint_every, 1) == 0):
            print(
                f"[LFD WP] frame={int(self.frame[0].item())} "
                f"wpt_idx={int(self.wpt_idx[0].item())} "
                f"goal=({goal_pos[0,0]:.3f},{goal_pos[0,1]:.3f},{goal_pos[0,2]:.3f}) "
                f"(wpt={int(self.wpt_idx[0].item())}) "
                f"ee=({ee_pos[0,0]:.3f},{ee_pos[0,1]:.3f},{ee_pos[0,2]:.3f})",
                flush=True,
            )
        if self._debug_joints and (self.frame[0].item() % max(self._debug_joints_every, 1) == 0):
            q0 = [float(v) for v in q[0].tolist()]
            qd0 = [float(v) for v in qd[0].tolist()]
            print(
                f"[LFD JOINTS] frame={int(self.frame[0].item())} "
                f"q={q0} qd={qd0}",
                flush=True,
            )
        if self._debug_obs and (self.frame[0].item() % max(self._debug_obs_every, 1) == 0):
            q0 = [float(v) for v in q[0].tolist()]
            qd0 = [float(v) for v in qd[0].tolist()]
            qdd0 = [float(v) for v in qdd[0].tolist()]
            tau0 = [float(v) for v in self.joint_actions[0].tolist()]
            ee0 = [float(v) for v in ee_pos[0].tolist()]
            goal0 = [float(v) for v in goal_pos[0].tolist()]
            print(
                f"[LFD OBS] frame={int(self.frame[0].item())} "
                f"q={q0} qd={qd0} qdd={qdd0} tau={tau0} "
                f"ee={ee0} goal={goal0} fz={float(self._fz_ema[0].item()):.3f} "
                f"phase={int(self.phase[0].item())} obs_dim={int(obs.shape[1])}",
                flush=True,
            )
        if self._debug_x and (self.frame[0].item() % max(self._debug_x_every, 1) == 0):
            ee_x = ee_pos[:, 0].detach().cpu().tolist()
            goal_x = goal_pos[:, 0].detach().cpu().tolist()
            wpt_idx = self.wpt_idx.detach().cpu().tolist()
            header = f"[LFD X] frame={int(self.frame[0].item())}"
            lines = [header]
            for idx in range(self.num_envs):
                dx = ee_x[idx] - goal_x[idx]
                lines.append(
                    f"  env{idx:02d}: ee_x={ee_x[idx]:.3f} "
                    f"goal_x({int(wpt_idx[idx])})={goal_x[idx]:.3f} dx={dx:+.3f}"
                )
            print("\n".join(lines), flush=True)

        if self._debug_discrete_post and self.num_envs > 0:
            self._discrete_debug_post_counter += 1
            if self._discrete_debug_post_counter % max(self._debug_discrete_post_every, 1) == 0:
                e = 0
                if self._prev_ee_valid[e]:
                    delta_ee = ee_pos[e] - self._prev_ee_pos[e]
                    cmd_err = self._last_cmd_pos[e] - ee_pos[e]
                    print(
                        f"[LFD DIS POST] step={int(self.frame[e].item())} "
                        f"ee=({ee_pos[e,0]:.3f},{ee_pos[e,1]:.3f},{ee_pos[e,2]:.3f}) "
                        f"d_ee=({delta_ee[0]:+.4f},{delta_ee[1]:+.4f},{delta_ee[2]:+.4f}) "
                        f"cmd=({self._last_cmd_pos[e,0]:.3f},{self._last_cmd_pos[e,1]:.3f},{self._last_cmd_pos[e,2]:.3f}) "
                        f"cmd_err=({cmd_err[0]:+.4f},{cmd_err[1]:+.4f},{cmd_err[2]:+.4f}) "
                        f"act={int(self._last_action_idx[e].item())} "
                        f"delta=({self._last_delta[e,0]:+.4f},{self._last_delta[e,1]:+.4f},{self._last_delta[e,2]:+.4f})",
                        flush=True,
                    )
            self._prev_ee_pos[:] = ee_pos
            self._prev_ee_valid[:] = True
        if self._debug_sanity and self.num_envs > 0:
            self._sanity_counter += 1
            if self._sanity_counter % max(self._debug_sanity_every, 1) == 0:
                e = 0
                cmd_err = self._last_cmd_pos[e] - ee_pos[e]
                cmd_err_norm = float(torch.linalg.vector_norm(cmd_err).item())
                goal_err = goal_pos[e] - ee_pos[e]
                goal_err_norm = float(torch.linalg.vector_norm(goal_err).item())
                prev_dist = float(self.prev_dist[e].item()) if hasattr(self, "prev_dist") else float("nan")
                print(
                    f"[LFD SANITY] frame={int(self.frame[e].item())} "
                    f"wpt_idx={int(self.wpt_idx[e].item())} phase={int(self.phase[e].item())} "
                    f"ee=({ee_pos[e,0]:.3f},{ee_pos[e,1]:.3f},{ee_pos[e,2]:.3f}) "
                    f"goal=({goal_pos[e,0]:.3f},{goal_pos[e,1]:.3f},{goal_pos[e,2]:.3f}) "
                    f"cmd=({self._last_cmd_pos[e,0]:.3f},{self._last_cmd_pos[e,1]:.3f},{self._last_cmd_pos[e,2]:.3f}) "
                    f"goal_err={goal_err_norm:.4f} cmd_err={cmd_err_norm:.4f} prev_dist={prev_dist:.4f} "
                    f"act={int(self._last_action_idx[e].item())} applied={int(self._last_applied_action_idx[e].item())} "
                    f"hold={int(self._last_hold[e].item())} mask={int(self._last_masked[e].item())}",
                    flush=True,
                )

        # RL-Games expects both policy and critic keys
        return {"policy": obs, "critic": obs}

    def _get_rewards(self) -> torch.Tensor:
        env_origins = self._get_env_origins()
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_idx]
        ee_pos = ee_pos_w - env_origins
        goal_pos, _ = self.traj_mgr.get_targets(torch.arange(self.num_envs, device=self.device), self.wpt_idx)
        reward_mode = self.reward_mode.lower()
        wpt_idx_before = self.wpt_idx.clone()
        prev_dist = self.prev_dist
        reward_kwargs = None
        hit = None
        w_dtau = 0.0
        p_dtau = None
        debug_r_track = None
        debug_r_progress = None
        debug_p_smooth = None
        debug_p_energy = None
        reward_components = {}

        if self.reward_fn is not None:
            # Use versioned reward function (wp_v1, wp_v2, or legacy wp)
            reward_kwargs = {
                "eps_in": _env_float("LFD_EPS_IN", 0.01),
                "sigma": _env_float("LFD_SIGMA", 0.03),
                "c_clip": _env_float("LFD_C_CLIP", 0.02),
                "w_wp": _env_float("LFD_W_WP", 1.0),
                "w_prog": _env_float("LFD_W_PROG", 2.0),
                "w_hit": _env_float("LFD_W_HIT", 5.0),
            }
            # Add time penalty weight for v2/v3/wp_x (ignored by v1 signature)
            if self.reward_mode.lower() in ("wp_v2", "wp_v3", "wp_v4_dis", "wp_x", "wp_x_only", "wp_y", "wp_y_only"):
                reward_kwargs["w_time"] = _env_float("LFD_W_TIME", -0.1)
            
            reward, new_wpt_idx, new_prev_dist, _hit = self.reward_fn(
                ee_pos,
                self.traj_mgr.p_traj_env,
                self.wpt_idx,
                self.prev_dist,
                **reward_kwargs
            )
            self.wpt_idx = new_wpt_idx
            self.prev_dist = new_prev_dist
            hit = _hit
            # Track waypoint advancement
            wpt_adv = torch.clamp(self.wpt_idx - wpt_idx_before, min=0)
            self._wpt_hits += wpt_adv
            self._wpt_max = torch.maximum(self._wpt_max, self.wpt_idx)
            # Reward component breakdown (weighted).
            eps_in = reward_kwargs["eps_in"]
            sigma = reward_kwargs["sigma"]
            c_clip = reward_kwargs["c_clip"]
            w_wp = reward_kwargs["w_wp"]
            w_prog = reward_kwargs["w_prog"]
            w_hit = reward_kwargs["w_hit"]
            w_time = reward_kwargs.get("w_time", 0.0)
            goal_before = self.traj_mgr.p_traj_env[torch.arange(self.num_envs, device=self.device), wpt_idx_before]
            delta = ee_pos - goal_before
            abs_delta = torch.abs(delta)
            if reward_mode in ("wp_x", "wp_x_only"):
                dist = abs_delta[:, 0]
                hit_dbg = dist < eps_in
            elif reward_mode in ("wp_y", "wp_y_only"):
                dist = abs_delta[:, 1]
                hit_dbg = dist < eps_in
            elif reward_mode in ("wp_v3", "wp_v4_dis"):
                axis_x = abs_delta[:, 0] >= eps_in
                axis_y = (~axis_x) & (abs_delta[:, 1] >= eps_in)
                dist = torch.where(axis_x, abs_delta[:, 0], torch.where(axis_y, abs_delta[:, 1], abs_delta[:, 2]))
                hit_dbg = (abs_delta[:, 0] < eps_in) & (abs_delta[:, 1] < eps_in) & (abs_delta[:, 2] < eps_in)
            else:
                dist = torch.linalg.vector_norm(delta, dim=1)
                hit_dbg = dist < eps_in
            if hit is not None:
                hit_dbg = hit
            r_wp = torch.exp(-((dist * dist) / (sigma * sigma)))
            r_prog = torch.clamp(prev_dist - dist, min=-c_clip, max=c_clip)
            r_hit = hit_dbg.float()
            r_time = torch.ones_like(dist) if w_time != 0.0 else torch.zeros_like(dist)
            reward_components = {
                "reward_total": reward,
                "c_wp": w_wp * r_wp,
                "c_prog": w_prog * r_prog,
                "c_hit": w_hit * r_hit,
                "c_time": w_time * r_time,
                "dist_active": dist,
                "hit": hit_dbg.float(),
            }
            # Penalize sharp torque changes (delta tau)
            w_dtau = _env_float("LFD_W_DTAU", 0.0)
            if w_dtau != 0.0:
                dtau = self.joint_actions - self.prev_joint_actions
                p_dtau = torch.linalg.vector_norm(dtau, dim=1)
                reward = reward - (w_dtau * p_dtau)
            reward_components["c_dtau"] = -w_dtau * p_dtau if p_dtau is not None else torch.zeros_like(reward)
            w_wpt_adv = _env_float("LFD_W_WPT_ADV", 0.0)
            if w_wpt_adv != 0.0:
                reward = reward + (wpt_adv.float() * w_wpt_adv)
            reward_components["c_wpt_adv"] = wpt_adv.float() * w_wpt_adv
            w_wpt_cover = _env_float("LFD_W_WPT_COVER", 0.0)
            if w_wpt_cover != 0.0:
                progress = self.wpt_idx.float() / max(self.traj_mgr.T - 1, 1)
                reward = reward + (w_wpt_cover * progress)
            else:
                progress = self.wpt_idx.float() / max(self.traj_mgr.T - 1, 1)
            reward_components["c_wpt_cover"] = w_wpt_cover * progress
            w_cmd_err = _env_float("LFD_W_CMD_ERR", 0.0)
            w_cmd_hit = _env_float("LFD_W_CMD_HIT", 0.0)
            if w_cmd_err != 0.0 or w_cmd_hit != 0.0:
                cmd_err = torch.linalg.vector_norm(self._last_cmd_pos - ee_pos, dim=1)
                if w_cmd_err != 0.0:
                    reward = reward - (w_cmd_err * cmd_err)
                if w_cmd_hit != 0.0:
                    cmd_eps = _env_float("LFD_CMD_EPS", float(self._discrete_eps))
                    reward = reward + (cmd_err <= cmd_eps).float() * w_cmd_hit
            else:
                cmd_err = torch.linalg.vector_norm(self._last_cmd_pos - ee_pos, dim=1)
                cmd_eps = _env_float("LFD_CMD_EPS", float(self._discrete_eps))
            reward_components["c_cmd_err"] = -w_cmd_err * cmd_err
            reward_components["c_cmd_hit"] = (cmd_err <= cmd_eps).float() * w_cmd_hit
            w_stuck = _env_float("LFD_W_STUCK", 0.0)
            if w_stuck != 0.0:
                stuck = self._stuck_count >= self._stuck_steps
                reward = reward - (w_stuck * stuck.float())
            else:
                stuck = self._stuck_count >= self._stuck_steps
            reward_components["c_stuck"] = -w_stuck * stuck.float()
        else:
            # Trajectory following reward focused on EE position tracking.
            pos_err = torch.linalg.vector_norm(goal_pos - ee_pos, dim=1)
            r_track = torch.exp(-pos_err / _env_float("LFD_POS_SCALE", 0.05))  # sharper when near goal
            progress = self.wpt_idx.float() / max(self.traj_mgr.T - 1, 1)
            r_progress = progress
            # Smoothness + energy penalties
            p_smooth = torch.linalg.vector_norm(self.joint_actions - self.prev_joint_actions, dim=1)
            p_energy = torch.linalg.vector_norm(self.joint_actions, dim=1)
            debug_r_track = r_track
            debug_r_progress = r_progress
            debug_p_smooth = p_smooth
            debug_p_energy = p_energy
            reward = (
                3.0 * r_track
                + 1.0 * r_progress
                - 0.01 * p_smooth
                - 0.001 * p_energy
            )
            reward_components = {
                "reward_total": reward,
                "c_track": 3.0 * r_track,
                "c_progress": 1.0 * r_progress,
                "c_smooth": -0.01 * p_smooth,
                "c_energy": -0.001 * p_energy,
            }
        # Success bonus at episode completion (trajectory-only by default).
        at_last = self.wpt_idx >= (self.traj_mgr.T - 1)
        if self._use_phase:
            if self._done_on_last:
                done_success = at_last
            else:
                done_success = (self.phase == 2) & at_last
        else:
            done_success = at_last
        if torch.any(done_success):
            w_done_success = _env_float("LFD_W_DONE_SUCCESS", 1_000_000.0)
            reward = reward + done_success.float() * w_done_success
        w_done_success = _env_float("LFD_W_DONE_SUCCESS", 1_000_000.0)
        reward_components["c_done_success"] = done_success.float() * w_done_success
        # Optional early-finish bonus: scale by remaining fraction of episode when done.
        w_done_early = _env_float("LFD_W_DONE_EARLY", 0.0)
        frac_left = None
        if w_done_early != 0.0 and torch.any(done_success):
            remaining = (int(self.max_episode_length) - 1) - self.episode_length_buf
            remaining = torch.clamp(remaining, min=0)
            frac_left = remaining.float() / max(int(self.max_episode_length) - 1, 1)
            reward = reward + done_success.float() * (w_done_early * frac_left)
        if frac_left is None:
            frac_left = torch.zeros_like(reward)
        reward_components["c_done_early"] = done_success.float() * (w_done_early * frac_left)
        reward_components["reward_total"] = reward
        # Update action history after reward computation
        self.prev_joint_actions = self.joint_actions.clone()

        # Cache scalar for logging
        try:
            self._r_last = float(reward[0].item())
        except Exception:
            self._r_last = 0.0
        # Attach reward components to per-step log (single env).
        if reward_components and self._traj_log_enabled and self._traj_log_env < self.num_envs:
            env_id = int(self._traj_log_env)
            if reward_components.get("reward_total") is not None:
                # Convert tensor components to floats for the selected env.
                comp = {}
                for key, val in reward_components.items():
                    try:
                        comp[key] = float(val[env_id].item())
                    except Exception:
                        try:
                            comp[key] = float(val)
                        except Exception:
                            pass
                self._update_reward_log(env_id, comp)

        # Reward debug: print breakdown for env0 (or all envs if enabled)
        if self._debug_reward and self.num_envs > 0:
            frame0 = int(self.frame[0].item()) if hasattr(self, "frame") else 0
            if frame0 % max(self._debug_reward_every, 1) == 0:
                done_timeout = self.episode_length_buf >= (int(self.max_episode_length) - 1)
                stuck = self._stuck_count >= self._stuck_steps
                header = (
                    f"[LFD REWARD] frame={frame0} mode={reward_mode} "
                    f"done={int(done_success.sum().item())} timeout={int(done_timeout.sum().item())} "
                    f"stuck={int(stuck.sum().item())}"
                )
                print(header, flush=True)
                env_ids = range(self.num_envs) if self._debug_reward_all else [0]
                if self.reward_fn is not None:
                    if reward_kwargs is None:
                        reward_kwargs = {
                            "eps_in": _env_float("LFD_EPS_IN", 0.01),
                            "sigma": _env_float("LFD_SIGMA", 0.03),
                            "c_clip": _env_float("LFD_C_CLIP", 0.02),
                            "w_wp": _env_float("LFD_W_WP", 1.0),
                            "w_prog": _env_float("LFD_W_PROG", 2.0),
                            "w_hit": _env_float("LFD_W_HIT", 5.0),
                        }
                        if reward_mode in ("wp_v2", "wp_v3", "wp_v4_dis", "wp_x", "wp_x_only", "wp_y", "wp_y_only"):
                            reward_kwargs["w_time"] = _env_float("LFD_W_TIME", -0.1)
                    eps_in = reward_kwargs["eps_in"]
                    sigma = reward_kwargs["sigma"]
                    c_clip = reward_kwargs["c_clip"]
                    w_wp = reward_kwargs["w_wp"]
                    w_prog = reward_kwargs["w_prog"]
                    w_hit = reward_kwargs["w_hit"]
                    w_time = reward_kwargs.get("w_time", 0.0)

                    delta = ee_pos - goal_pos
                    abs_delta = torch.abs(delta)
                    if reward_mode in ("wp_x", "wp_x_only"):
                        dist = abs_delta[:, 0]
                        hit_dbg = dist < eps_in
                    elif reward_mode in ("wp_y", "wp_y_only"):
                        dist = abs_delta[:, 1]
                        hit_dbg = dist < eps_in
                    elif reward_mode in ("wp_v3", "wp_v4_dis"):
                        axis_x = abs_delta[:, 0] >= eps_in
                        axis_y = (~axis_x) & (abs_delta[:, 1] >= eps_in)
                        dist = torch.where(axis_x, abs_delta[:, 0], torch.where(axis_y, abs_delta[:, 1], abs_delta[:, 2]))
                        hit_dbg = (abs_delta[:, 0] < eps_in) & (abs_delta[:, 1] < eps_in) & (abs_delta[:, 2] < eps_in)
                    else:
                        dist = torch.linalg.vector_norm(delta, dim=1)
                        hit_dbg = dist < eps_in
                    if hit is not None:
                        hit_dbg = hit
                    r_wp = torch.exp(-((dist * dist) / (sigma * sigma)))
                    r_prog = torch.clamp(prev_dist - dist, min=-c_clip, max=c_clip)
                    r_hit = hit_dbg.float()
                    r_time = torch.ones_like(dist) if w_time != 0.0 else torch.zeros_like(dist)
                    r_dtau = torch.zeros_like(dist)
                    if w_dtau != 0.0 and p_dtau is not None:
                        r_dtau = -w_dtau * p_dtau
                    for i in env_ids:
                        ep_len = int(self.episode_length_buf[i].item())
                        phase_i = int(self.phase[i].item()) if hasattr(self, "phase") else -1
                        w_before = int(wpt_idx_before[i].item())
                        w_after = int(self.wpt_idx[i].item())
                        dx = abs_delta[i, 0].item()
                        dy = abs_delta[i, 1].item()
                        dz = abs_delta[i, 2].item()
                        print(
                            f"  env{i:02d}: |dx|={dx:.4f} |dy|={dy:.4f} |dz|={dz:.4f} eps_in={eps_in:.3f}",
                            flush=True,
                        )
                        print(
                            f"  env{i:02d}: len={ep_len} phase={phase_i} wpt={w_before}->{w_after} dist={dist[i]:.4f} "
                            f"r_wp={r_wp[i]:.3f} r_prog={r_prog[i]:+.3f} r_hit={r_hit[i]:.1f} "
                            f"r_time={r_time[i]:+.2f} r_dtau={r_dtau[i]:+.3f} total={reward[i]:+.3f}",
                            flush=True,
                        )
                else:
                    pos_err = torch.linalg.vector_norm(goal_pos - ee_pos, dim=1)
                    r_track = debug_r_track if debug_r_track is not None else torch.exp(-pos_err / _env_float("LFD_POS_SCALE", 0.05))
                    r_progress = debug_r_progress if debug_r_progress is not None else (wpt_idx_before.float() / max(self.traj_mgr.T - 1, 1))
                    p_smooth = debug_p_smooth if debug_p_smooth is not None else torch.linalg.vector_norm(
                        self.joint_actions - self.prev_joint_actions, dim=1
                    )
                    p_energy = debug_p_energy if debug_p_energy is not None else torch.linalg.vector_norm(self.joint_actions, dim=1)
                    for i in env_ids:
                        ep_len = int(self.episode_length_buf[i].item())
                        phase_i = int(self.phase[i].item()) if hasattr(self, "phase") else -1
                        print(
                            f"  env{i:02d}: len={ep_len} phase={phase_i} pos_err={pos_err[i]:.4f} r_track={r_track[i]:.3f} "
                            f"r_prog={r_progress[i]:.3f} p_smooth={p_smooth[i]:.3f} p_energy={p_energy[i]:.3f} "
                            f"total={reward[i]:+.3f}",
                            flush=True,
                        )

        # Progress debug on env 0
        if self._debug_progress and self.num_envs > 0:
            frame0 = int(self.frame[0].item()) if hasattr(self, "frame") else 0
            if reward_mode in ("wp_x", "wp_x_only"):
                dist0 = float(torch.abs(goal_pos[0, 0] - ee_pos[0, 0]).item())
            elif reward_mode in ("wp_y", "wp_y_only"):
                dist0 = float(torch.abs(goal_pos[0, 1] - ee_pos[0, 1]).item())
            else:
                dist0 = torch.linalg.vector_norm(goal_pos[0] - ee_pos[0], dim=0).item()
            hit0 = bool(hit[0].item()) if hit is not None and len(hit) > 0 else False
            progressed = int(self.wpt_idx[0].item()) != self._last_wpt_idx_progress
            if (frame0 % max(self._debug_progress_every, 1) == 0) or hit0 or progressed:
                print(
                    f"[LFD PROG] frame={frame0} wpt_idx={int(self.wpt_idx[0].item())}/{self.traj_mgr.T-1} "
                    f"dist={dist0:.4f} hit={hit0} reward={self._r_last:.4f}",
                    flush=True,
                )
                self._last_wpt_idx_progress = int(self.wpt_idx[0].item())
        return reward

    def _get_dones(self):
        # Episode done if last waypoint is reached (optionally bypass phase) or timeout.
        at_last = self.wpt_idx >= (self.traj_mgr.T - 1)
        if self._use_phase:
            if self._done_on_last:
                done_success = at_last
            else:
                done_success = (self.phase == 2) & at_last
        else:
            done_success = at_last
        # Timeout
        max_steps = int(self.max_episode_length)
        done_timeout = self.episode_length_buf >= (max_steps - 1)
        if self._disable_stuck:
            stuck = torch.zeros_like(done_timeout, dtype=torch.bool)
        else:
            stuck = self._stuck_count >= self._stuck_steps
        terminated = done_success
        truncated = done_timeout | stuck
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        # Dump single-env JSON on reset if enabled and this reset includes the selected env
        if self._traj_log_enabled and env_ids.numel() > 0 and (self._traj_log_env in env_ids.tolist()):
            self._flush_traj_log(force=True)

        # Reset internal buffers
        self.wpt_idx[env_ids] = 0
        self.phase[env_ids] = 0
        self._fz_ema[env_ids] = 0.0
        self.prev_joint_actions[env_ids] = 0.0
        self.obs_buffer[env_ids] = 0.0
        self.frame[env_ids] = 0
        self._stuck_count[env_ids] = 0
        self._wpt_max[env_ids] = 0
        self._wpt_hits[env_ids] = 0
        self._wpt_step_count[env_ids] = 0
        self.joint_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0
        self._prev_ee_pos[env_ids] = 0.0
        self._prev_ee_valid[env_ids] = False
        self._cmd_pos_valid[env_ids] = False
        self._last_hold[env_ids] = False
        self._last_masked[env_ids] = False
        self._last_applied_action_idx[env_ids] = 0
        self._last_delta_raw[env_ids] = 0.0

        # Resample trajectory randomization
        env_ids_t = env_ids.to(self.device).to(torch.int64)
        self.traj_mgr.sample(env_ids_t)
        self._apply_traj_noise(env_ids_t)
        self._update_traj_bounds(env_ids_t)
        self._sample_tau_scale_env(env_ids_t)

        super()._reset_idx(env_ids)

        # Force robot to home pose to avoid inheriting previous episode pose
        try:
            # API 2.0.1: write_joint_state_to_sim expects full DOF vectors (pos, vel)
            if hasattr(self.robot.data, "default_joint_pos"):
                home_full = self.robot.data.default_joint_pos[env_ids]
            else:
                home_full = torch.zeros((len(env_ids), self.robot.num_dof), device=self.device)
            # Add optional Gaussian noise on initial joint pose
            if self.init_joint_noise_std > 0:
                home_full = home_full + torch.randn_like(home_full) * self.init_joint_noise_std
            if self._j1_offset_rad != 0.0:
                j1_idx = self.joint_ids[0]
                if hasattr(j1_idx, "item"):
                    j1_idx = int(j1_idx.item())
                else:
                    j1_idx = int(j1_idx)
                home_full[:, j1_idx] = home_full[:, j1_idx] + self._j1_offset_rad
            # Add optional Gaussian noise on initial joint velocities
            if self.init_joint_vel_std > 0:
                vel_full = torch.randn_like(home_full) * self.init_joint_vel_std
            else:
                vel_full = torch.zeros_like(home_full)
            if hasattr(self.robot, "write_joint_state_to_sim"):
                self.robot.write_joint_state_to_sim(home_full, vel_full)
        except Exception as exc:
            if self._debug_reset:
                print(f"[LFD RESET] failed to force home pose: {exc}", flush=True)

        # Optional TCP offset after home pose
        if torch.any(self._tcp_offset != 0):
            try:
                env_origins = self._get_env_origins()
                ee_pos_w = self.robot.data.body_pos_w[env_ids, self.ee_body_idx]
                ee_quat = self.robot.data.body_quat_w[env_ids, self.ee_body_idx]
                offset = self._tcp_offset.to(self.device)
                if offset.ndim == 1:
                    offset = offset.unsqueeze(0)
                offset = offset.expand(len(env_ids), -1)
                target_pos_w = ee_pos_w + offset
                cmd = torch.cat((target_pos_w, ee_quat), dim=1)
                self.diffik.set_command(cmd, ee_pos=ee_pos_w, ee_quat=ee_quat)
                jac_w = self.robot.root_physx_view.get_jacobians()[env_ids, self.ee_jac_idx, :, self.joint_ids]
                q_des = self.diffik.compute(
                    ee_pos_w, ee_quat, jac_w, self.robot.data.joint_pos[env_ids, self.joint_ids]
                )
                if hasattr(self.robot, "write_joint_state_to_sim"):
                    full_q = self.robot.data.joint_pos[env_ids].clone()
                    full_q[:, self.joint_ids] = q_des
                    full_qd = torch.zeros_like(full_q)
                    self.robot.write_joint_state_to_sim(full_q, full_qd)
            except Exception as exc:
                if self._debug_reset:
                    print(f"[LFD RESET] tcp offset failed: {exc}", flush=True)

        # Debug: print pose after reset (only for selected env) to ensure home reset works
        if self._debug_reset and env_ids.numel() > 0:
            e = env_ids[0]
            try:
                env_origins = self._get_env_origins()
                ee_pos = (self.robot.data.body_pos_w[e, self.ee_body_idx] - env_origins[e]).detach().cpu().tolist()
                joint_pos = self.robot.data.joint_pos[e, self.joint_ids].detach().cpu().tolist()
                base_p0 = self.traj_mgr.p_traj[0].detach().cpu()
                env_p0 = self.traj_mgr.p_traj_env[e, 0].detach().cpu()
                offset = (env_p0 - base_p0).tolist()
                tcp_offset = self._tcp_offset.detach().cpu().tolist()
                print(
                    f"[LFD RESET] env={int(e)} wpt_idx={int(self.wpt_idx[e].item())} "
                    f"ee_pos=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}) "
                    f"joint_pos0={joint_pos[0]:.3f} "
                    f"traj_offset=({offset[0]:+.4f},{offset[1]:+.4f},{offset[2]:+.4f}) "
                    f"tcp_offset=({tcp_offset[0]:+.4f},{tcp_offset[1]:+.4f},{tcp_offset[2]:+.4f}) "
                    f"j1_offset_deg={self._j1_offset_deg:.2f}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[LFD RESET] failed to read pose after reset: {exc}", flush=True)

        # Initialize command position to the current EE pose after reset.
        try:
            env_origins = self._get_env_origins()
            ee_pos = self.robot.data.body_pos_w[env_ids, self.ee_body_idx] - env_origins[env_ids]
            self._cmd_pos[env_ids] = ee_pos
            self._cmd_pos_valid[env_ids] = True
            self._last_cmd_pos[env_ids] = ee_pos
            self._last_action_idx[env_ids] = 0
            self._last_applied_action_idx[env_ids] = 0
            self._last_delta_raw[env_ids] = 0.0
            self._last_delta[env_ids] = 0.0
        except Exception:
            self._cmd_pos_valid[env_ids] = False

        # Reset prev_dist to current distance to first waypoint
        try:
            env_ids_t = env_ids.to(self.device)
            goals0 = self.traj_mgr.p_traj_env[env_ids_t, self.wpt_idx[env_ids_t]]
            env_origins = self._get_env_origins()
            ee_pos = self.robot.data.body_pos_w[env_ids_t, self.ee_body_idx] - env_origins[env_ids_t]
            mode = self.reward_mode.lower()
            if mode in ("wp_x", "wp_x_only"):
                dist0 = torch.abs(ee_pos[:, 0] - goals0[:, 0])
            elif mode in ("wp_v3", "wp_v4_dis"):
                delta = torch.abs(ee_pos - goals0)
                eps_in = _env_float("LFD_EPS_IN", 0.01)
                axis_x = delta[:, 0] >= eps_in
                axis_y = (~axis_x) & (delta[:, 1] >= eps_in)
                dist0 = torch.where(axis_x, delta[:, 0], torch.where(axis_y, delta[:, 1], delta[:, 2]))
            else:
                dist0 = torch.linalg.norm(ee_pos - goals0, dim=1)
            self.prev_dist[env_ids_t] = dist0
        except Exception:
            self.prev_dist[env_ids] = 0.0

    def export_log_dict(self) -> dict:
        """Minimal log dict to satisfy EpisodeLogger in play.py."""
        # Step/time
        step = int(self.frame[0].item()) if hasattr(self, "frame") else 0
        dt = float(getattr(self, "physics_dt", getattr(self.cfg.sim, "dt", 1 / 120)))
        t = step * dt

        # Goal / current EE pose
        try:
            goal_pos_w, _ = self.traj_mgr.get_targets(torch.tensor([0], device=self.device), self.wpt_idx[:1])
            p_des = goal_pos_w[0].detach().cpu().tolist()
        except Exception:
            p_des = [0.0, 0.0, 0.0]

        try:
            env_origins = self._get_env_origins()
            p_act = (self.robot.data.body_pos_w[0, self.ee_body_idx] - env_origins[0]).detach().cpu().tolist()
        except Exception:
            p_act = [0.0, 0.0, 0.0]

        # Joint states
        try:
            joint_pos = self.robot.data.joint_pos[0, self.joint_ids].detach().cpu().tolist()
            joint_vel = self.robot.data.joint_vel[0, self.joint_ids].detach().cpu().tolist()
        except Exception:
            joint_pos = [0.0] * 7
            joint_vel = [0.0] * 7

        # Applied torques
        try:
            tau_app = self.joint_actions[0].detach().cpu().tolist()
        except Exception:
            tau_app = [0.0] * 7

        # Force (EMA on Z)
        try:
            fz = float(self._fz_ema[0].item())
        except Exception:
            fz = float("nan")

        phase_val = int(self.phase[0].item()) if hasattr(self, "phase") else -1

        return {
            "step": step,
            "time": t,
            "p_des": p_des,
            "p_act": p_act,
            "q_des": [0.0] * 7,
            "q_act": [0.0] * 7,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "tau_comp": [0.0] * 7,
            "tau_app": tau_app,
            "f_ext": [0.0, 0.0, fz],
            "manip": float("nan"),
            "pos_err": float("nan"),
            "ori_err": float("nan"),
            "kp": [0.0],
            "zeta": [0.0],
            "reward": self._r_last,
            "phase": phase_val,
            "wpt_idx": int(self.wpt_idx[0].item()) if hasattr(self, "wpt_idx") else -1,
        }

    def close(self):
        # Flush any pending log if the sim shuts down before a reset
        if self._traj_log_enabled and self._traj_log_env < self.num_envs:
            self._flush_traj_log(force=False)
        super().close()
