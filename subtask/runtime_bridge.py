#!/usr/bin/env python3
"""
Shared runtime helpers for subtask execution.

Subtasks should focus on target generation and solver calls. Joint-state
application, MuJoCo forward updates, end-effector pose reads, viewer sync, and
optional command emission are centralized here.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional, Tuple

import mujoco
import numpy as np


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
END_EFFECTOR_SITE = "attachment_site"


def joint_qpos_addresses(model_mj: mujoco.MjModel):
    return [
        model_mj.jnt_qposadr[mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in JOINT_NAMES
    ]


def read_joint_positions(model_mj: mujoco.MjModel, data_mj: mujoco.MjData) -> np.ndarray:
    return np.array([data_mj.qpos[idx] for idx in joint_qpos_addresses(model_mj)], dtype=float)


def joint_actuator_ids(model_mj: mujoco.MjModel):
    actuator_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    return [
        mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in actuator_names
    ]


def apply_joint_positions(model_mj: mujoco.MjModel, data_mj: mujoco.MjData, joint_angles: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(joint_angles), dtype=float).reshape(-1)
    if values.size != len(JOINT_NAMES):
        raise ValueError(f"Expected {len(JOINT_NAMES)} joint angles, got {values.size}.")

    for idx, val in zip(joint_qpos_addresses(model_mj), values):
        data_mj.qpos[idx] = float(val)
    return values


def apply_joint_controls(model_mj: mujoco.MjModel, data_mj: mujoco.MjData, joint_angles: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(joint_angles), dtype=float).reshape(-1)
    if values.size != len(JOINT_NAMES):
        raise ValueError(f"Expected {len(JOINT_NAMES)} joint angles, got {values.size}.")

    for actuator_id, val in zip(joint_actuator_ids(model_mj), values):
        if actuator_id >= 0:
            data_mj.ctrl[actuator_id] = float(val)
    return values


def emit_motion_step(solver, joint_angles: Iterable[float], stage: Optional[str] = None):
    if solver is None or stage is None:
        return None

    callback = getattr(solver, "on_joint_state_applied", None)
    if callable(callback):
        return callback(joint_angles, stage=stage)

    legacy_emit = getattr(solver, "emit_command", None)
    if callable(legacy_emit):
        return legacy_emit(joint_angles, stage=stage)
    return None


def notify_scene_state_applied(
    solver,
    model_mj: mujoco.MjModel,
    data_mj: mujoco.MjData,
    joint_angles: Iterable[float],
    stage: Optional[str] = None,
):
    if solver is None:
        return None

    callback = getattr(solver, "on_scene_state_applied", None)
    if callable(callback):
        return callback(model_mj, data_mj, joint_angles, stage=stage)
    return None


def apply_motion_step(
    solver,
    model_mj: mujoco.MjModel,
    data_mj: mujoco.MjData,
    joint_angles: Iterable[float],
    stage: Optional[str] = None,
    viewer=None,
    delay: float = 0.0,
    physics_steps: int = 0,
    kinematic: bool = True,
) -> np.ndarray:
    target_values = np.asarray(list(joint_angles), dtype=float).reshape(-1)
    if target_values.size != len(JOINT_NAMES):
        raise ValueError(f"Expected {len(JOINT_NAMES)} joint angles, got {target_values.size}.")

    if kinematic:
        apply_joint_positions(model_mj, data_mj, target_values)
    apply_joint_controls(model_mj, data_mj, target_values)
    mujoco.mj_forward(model_mj, data_mj)
    emit_motion_step(solver, target_values, stage=stage)

    step_count = max(0, int(physics_steps))
    if not kinematic and step_count == 0:
        step_count = 1
    for _ in range(step_count):
        mujoco.mj_step(model_mj, data_mj)

    values = read_joint_positions(model_mj, data_mj)
    notify_scene_state_applied(solver, model_mj, data_mj, values, stage=stage)

    if viewer:
        viewer.sync()
        if delay > 0:
            time.sleep(delay)

    return values


def get_end_effector_pose(model_mj: mujoco.MjModel, data_mj: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
    site_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITE)
    pos = data_mj.site_xpos[site_id].copy()
    rot = data_mj.site_xmat[site_id].flatten()
    return pos, rot
