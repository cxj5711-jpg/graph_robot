#!/usr/bin/env python3
"""
Subtask: orientation adjustment.
"""

import numpy as np

from subtask.runtime_bridge import apply_motion_step, get_end_effector_pose
from subtask import task_linear


TARGET_DIRECTION = [1, 1, -1]


def _direction_to_rotation_matrix(direction):
    z_axis = np.array(direction, dtype=float)
    norm = np.linalg.norm(z_axis)
    if norm < 1e-6:
        raise ValueError("Input direction cannot be a zero vector.")
    z_axis = z_axis / norm

    if abs(z_axis[2]) < 0.9:
        temp = np.array([0.0, 0.0, 1.0])
    else:
        temp = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(temp, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    return np.array([
        x_axis[0], y_axis[0], z_axis[0],
        x_axis[1], y_axis[1], z_axis[1],
        x_axis[2], y_axis[2], z_axis[2],
    ])


def _rotation_matrix_to_quaternion(rot_matrix):
    rot = np.asarray(rot_matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(rot))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qw, qx, qy, qz], dtype=float)
    return quat / np.linalg.norm(quat)


def _quaternion_to_rotation_matrix(quat):
    q = np.asarray(quat, dtype=float).reshape(4)
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q

    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=float)


def _slerp(quat_start, quat_end, t):
    q0 = np.asarray(quat_start, dtype=float).reshape(4)
    q1 = np.asarray(quat_end, dtype=float).reshape(4)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        blended = q0 + t * (q1 - q0)
        return blended / np.linalg.norm(blended)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    return s0 * q0 + s1 * q1


def execute(solver, model_mj, data_mj, viewer, current_angles, current_pos, current_rot, target_xyz):
    print(f"[Task 4] Adjusting orientation toward {TARGET_DIRECTION}...")
    target_rot = _direction_to_rotation_matrix(TARGET_DIRECTION).reshape(3, 3)
    current_rot_matrix = np.asarray(current_rot, dtype=float).reshape(3, 3)

    steps = max(1, int(task_linear.INTERPOLATION_STEPS))
    delay = max(0.0, float(task_linear.DELAY_PER_STEP))
    start_quat = _rotation_matrix_to_quaternion(current_rot_matrix)
    target_quat = _rotation_matrix_to_quaternion(target_rot)

    angles = current_angles.copy()
    success_all = True

    for i in range(1, steps + 1):
        t = i / steps
        interp_rot = _quaternion_to_rotation_matrix(_slerp(start_quat, target_quat, t)).flatten()

        angles, success = solver.solve(
            angles,
            current_pos,
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
            delay=delay,
        )

    if not success_all:
        print("[Task 4] Orientation interpolation encountered at least one IK failure.")

    final_pos, final_rot = get_end_effector_pose(model_mj, data_mj)
    return angles, final_pos, final_rot, success_all
