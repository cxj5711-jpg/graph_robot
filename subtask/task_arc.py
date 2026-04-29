#!/usr/bin/env python3
"""
Subtask: arc motion.
"""

import numpy as np

from subtask.runtime_bridge import apply_motion_step, get_end_effector_pose


ARC_CENTER = [0.0, 0.0, 0.0]
INTERPOLATION_STEPS = 30
DELAY_PER_STEP = 0.02


def _get_rotation_matrix_around_axis(axis, angle):
    k = axis / np.linalg.norm(axis)
    kx, ky, kz = k
    skew = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0],
    ])
    eye = np.eye(3)
    return eye + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def execute(solver, model_mj, data_mj, viewer, current_angles, current_pos, current_rot, target_xyz):
    center = np.array(ARC_CENTER, dtype=float)
    start_pos = current_pos
    end_pos = target_xyz

    v_start = start_pos - center
    v_end = end_pos - center
    radius = np.linalg.norm(v_start)

    normal = np.cross(v_start, v_end)
    norm_val = np.linalg.norm(normal)
    if norm_val < 1e-6:
        raise ValueError("Start point, end point, and arc center are collinear.")
    normal = normal / norm_val

    cos_theta = np.clip(np.dot(v_start, v_end) / (radius * np.linalg.norm(v_end)), -1.0, 1.0)
    arc_angle_rad = np.arccos(cos_theta)

    print(f"[Task 3] Executing arc scan around center {np.round(center, 3)}...")
    print(f"         > Plane normal: {np.round(normal, 3)}")
    print(f"         > Arc span: {np.degrees(arc_angle_rad):.1f} deg")

    angles = current_angles.copy()
    success_all = True

    u_axis = v_start / radius
    v_axis = np.cross(normal, u_axis)
    init_rot_matrix = current_rot.reshape((3, 3))

    for i in range(1, INTERPOLATION_STEPS + 1):
        t = i / INTERPOLATION_STEPS
        current_angle = t * arc_angle_rad

        interp_pos = center + radius * (np.cos(current_angle) * u_axis + np.sin(current_angle) * v_axis)
        rot_offset = _get_rotation_matrix_around_axis(normal, current_angle)
        interp_rot = (rot_offset @ init_rot_matrix).flatten()

        angles, success = solver.solve(
            angles,
            interp_pos,
            interp_rot,
            verbose=False,
            emit_solver_iteration=False,
            emit_solver_final=False,
        )
        if not success:
            success_all = False

        apply_motion_step(
            solver,
            model_mj,
            data_mj,
            angles,
            stage="linear_step",
            viewer=viewer,
            delay=DELAY_PER_STEP,
        )

    final_pos, final_rot = get_end_effector_pose(model_mj, data_mj)
    return angles, final_pos, final_rot, success_all
