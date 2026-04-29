#!/usr/bin/env python3
"""
Subtask: screw-in assembly.
Helical motion is executed step-by-step:
1) feed along assembly axis
2) rotate around the same axis
"""

import time

import numpy as np

from subtask.runtime_bridge import apply_motion_step, get_end_effector_pose


TARGET_DIRECTION = [1, 1, -1]

SCREW_TURNS = 1.0
SCREW_DEPTH = 0.03
SCREW_STEPS = 40
DELAY_PER_STEP = 0.05
SCREW_FEED_WITH_TARGET_DIRECTION = True
REMOTE_SETTLE_TIME = 0.10

SOLVER_MAX_ITER = 60
SOLVER_TOL_POS = 0.008
SOLVER_TOL_ROT = 0.01
SOLVER_STEP_CLIP = 0.2

MAX_FAIL_RATIO = 0.10
HARD_FAIL_RATIO = 0.95
FINAL_POS_TOL = 0.012
FINAL_ROT_MSE_TOL = 0.02


def _normalize(vec):
    arr = np.array(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm < 1e-6:
        raise ValueError("TARGET_DIRECTION cannot be a zero vector.")
    return arr / norm


def _rotation_matrix(axis, angle):
    axis = _normalize(axis)
    kx, ky, kz = axis
    skew = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0],
    ])
    eye = np.eye(3)
    return eye + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _get_command_client(solver):
    bridge = getattr(solver, "motion_bridge", None)
    if bridge is not None:
        return bridge
    return getattr(solver, "command_client", None)


def _sleep_with_viewer(viewer, duration: float, dt: float = 0.02):
    steps = max(1, int(duration / dt))
    for _ in range(steps):
        if viewer:
            viewer.sync()
        if dt > 0:
            time.sleep(dt)


def execute(solver, model_mj, data_mj, viewer, current_angles, current_pos, current_rot, target_xyz):
    axis_direction = _normalize(TARGET_DIRECTION)
    if not SCREW_FEED_WITH_TARGET_DIRECTION:
        axis_direction = -axis_direction

    steps = max(1, int(SCREW_STEPS))
    total_rotation = 2.0 * np.pi * float(SCREW_TURNS)
    delta_rotation = total_rotation / steps

    print(f"[Task 6] Screw-in start: turns={SCREW_TURNS}, depth={SCREW_DEPTH}m, steps={steps}")

    start_pos = current_pos.copy()
    start_rot = current_rot.reshape(3, 3).copy()

    ik_angles = current_angles.copy()
    wrist_acc = float(current_angles[5])
    wrist_start = wrist_acc

    fail_steps = 0
    curr_pos = current_pos.copy()
    curr_rot = current_rot.copy()
    cmd_angles = current_angles.copy()
    command_client = _get_command_client(solver)
    batch_remote_program = (
        command_client is not None
        and getattr(command_client, "enabled", False)
        and hasattr(command_client, "send_program")
        and callable(command_client.send_program)
        and (
            not hasattr(command_client, "stage_enabled")
            or command_client.stage_enabled("linear_step")
        )
    )
    planned_joint_steps = []
    remote_commands = []

    for step in range(1, steps + 1):
        progress = step / steps
        theta = total_rotation * progress

        interp_pos = start_pos + axis_direction * (float(SCREW_DEPTH) * progress)
        rot_offset = _rotation_matrix(axis_direction, theta)
        interp_rot = (rot_offset @ start_rot).flatten()

        solved_angles, success = solver.solve(
            ik_angles,
            interp_pos,
            interp_rot,
            max_iter=SOLVER_MAX_ITER,
            tol_pos=SOLVER_TOL_POS,
            tol_rot=SOLVER_TOL_ROT,
            step_clip=SOLVER_STEP_CLIP,
            verbose=False,
            emit_solver_iteration=False,
            emit_solver_final=False,
        )
        if not success:
            fail_steps += 1
        ik_angles = solved_angles.copy()

        wrist_acc += delta_rotation
        cmd_angles = solved_angles.copy()
        cmd_angles[5] = wrist_acc
        planned_joint_steps.append(cmd_angles.copy())

        if batch_remote_program:
            remote_commands.append(
                command_client.format_movej(
                    cmd_angles,
                    t=DELAY_PER_STEP,
                    r=0.0,
                )
            )
        else:
            apply_motion_step(
                solver,
                model_mj,
                data_mj,
                cmd_angles,
                stage="linear_step",
                viewer=viewer,
                delay=DELAY_PER_STEP,
            )
            curr_pos, curr_rot = get_end_effector_pose(model_mj, data_mj)

    if batch_remote_program and remote_commands:
        command_client.send_program(remote_commands, program_name="task3_screw_sequence")
        for cmd_step in planned_joint_steps:
            apply_motion_step(
                solver,
                model_mj,
                data_mj,
                cmd_step,
                stage=None,
                viewer=viewer,
                delay=DELAY_PER_STEP,
            )
            curr_pos, curr_rot = get_end_effector_pose(model_mj, data_mj)
        _sleep_with_viewer(viewer, REMOTE_SETTLE_TIME)

    actual_turns = (wrist_acc - wrist_start) / (2.0 * np.pi)
    fail_ratio = fail_steps / steps

    final_theta = total_rotation
    final_target_pos = start_pos + axis_direction * float(SCREW_DEPTH)
    final_rot_offset = _rotation_matrix(axis_direction, final_theta)
    final_target_rot = (final_rot_offset @ start_rot).flatten()
    final_pos_err = np.linalg.norm(curr_pos - final_target_pos)
    final_rot_err = np.mean(np.square(curr_rot - final_target_rot))

    success_by_ratio = fail_ratio <= MAX_FAIL_RATIO
    success_by_hard_ratio = fail_ratio <= HARD_FAIL_RATIO
    success_by_final = final_pos_err <= FINAL_POS_TOL and final_rot_err <= FINAL_ROT_MSE_TOL
    success_final = success_by_final and success_by_hard_ratio

    print(
        f"[Task 6] Screw-in done: commanded turns={actual_turns:.3f}, "
        f"fail_steps={fail_steps}/{steps}, "
        f"final_pos_err={final_pos_err * 1000.0:.2f}mm, final_rot_mse={final_rot_err:.5f}"
    )
    if not success_by_ratio:
        print(
            f"[Task 6] Warning: fail ratio {fail_ratio:.1%} exceeds monitor threshold "
            f"{MAX_FAIL_RATIO:.1%}, but final pose is {'OK' if success_by_final else 'NOT OK'}."
        )

    return cmd_angles, curr_pos, curr_rot, success_final
