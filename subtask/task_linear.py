#!/usr/bin/env python3
"""
Subtask: linear motion.
"""

import numpy as np

from subtask.runtime_bridge import apply_motion_step, get_end_effector_pose


INTERPOLATION_STEPS = 20
DELAY_PER_STEP = 0.02
PHYSICS_STEPS_PER_MOTION_STEP = 0
PHYSICS_DRIVEN_MOTION = False


def execute(solver, model_mj, data_mj, viewer, current_angles, current_pos, current_rot, target_xyz):
    print("[Task 2] Executing linear interpolation...")
    angles = current_angles.copy()
    success_all = True

    for i in range(1, INTERPOLATION_STEPS + 1):
        t = i / INTERPOLATION_STEPS
        interp_pos = current_pos + t * (target_xyz - current_pos)

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
            delay=DELAY_PER_STEP,
            physics_steps=PHYSICS_STEPS_PER_MOTION_STEP,
            kinematic=not PHYSICS_DRIVEN_MOTION,
        )

    final_pos, final_rot = get_end_effector_pose(model_mj, data_mj)
    return angles, final_pos, final_rot, success_all
