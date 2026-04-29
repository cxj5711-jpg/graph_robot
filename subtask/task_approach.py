#!/usr/bin/env python3
"""
Subtask: approach.
"""

import numpy as np

from subtask.runtime_bridge import apply_motion_step, get_end_effector_pose
from subtask import task_linear
from subtask.task_orient import TARGET_DIRECTION


APPROACH_DISTANCE = 0.3


def execute(solver, model_mj, data_mj, viewer, current_angles, current_pos, current_rot, target_xyz):
    axis_vector = np.array(TARGET_DIRECTION, dtype=float)
    norm = np.linalg.norm(axis_vector)
    if norm < 1e-6:
        raise ValueError("TARGET_DIRECTION cannot be a zero vector.")
    axis_direction = axis_vector / norm

    print(f"[Task 1] Approaching along direction {TARGET_DIRECTION}...")
    approach_pos = target_xyz - APPROACH_DISTANCE * axis_direction

    steps = max(1, int(task_linear.INTERPOLATION_STEPS))
    delay = max(0.0, float(task_linear.DELAY_PER_STEP))

    angles = current_angles.copy()
    success_all = True

    for i in range(1, steps + 1):
        t = i / steps
        interp_pos = current_pos + t * (approach_pos - current_pos)

        angles, success = solver.solve(
            angles,
            interp_pos,
            current_rot,
            verbose=False,
            emit_solver_iteration=False,
            emit_solver_final=False,
        )
        if not success:
            success_all = False

        angles = apply_motion_step(
            solver,
            model_mj,
            data_mj,
            angles,
            stage="linear_step",
            viewer=viewer,
            delay=delay,
            physics_steps=getattr(task_linear, "PHYSICS_STEPS_PER_MOTION_STEP", 0),
            kinematic=not getattr(task_linear, "PHYSICS_DRIVEN_MOTION", False),
        )

    final_pos, final_rot = get_end_effector_pose(model_mj, data_mj)
    return angles, final_pos, final_rot, success_all
