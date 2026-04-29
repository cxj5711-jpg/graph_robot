#!/usr/bin/env python3
"""Behavior-tree demo for visual inspection using the vision target."""

from __future__ import annotations

import os
import sys
import time
from enum import Enum

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from GNN.ik_solver_api import GNNIKSolver
from robot_ip_client import RobotIPClient
from run_vision import XML_PATH, ViewerWithCameraInset, prepare_vision
from subtask import task_approach
from subtask import task_arc
from subtask import task_hover
from subtask import task_linear
from subtask import task_orient
from subtask import task_reach
from subtask.runtime_bridge import apply_motion_step, get_end_effector_pose

# =========================
# Task 2 control parameters
# =========================
# 任务开始时的 6 关节初始角（弧度）。
INITIAL_ANGLES = np.array([3.14, -1.57, 1.57, 1.57, 1.57, 0.0], dtype=float)
# 相对视觉目标点的扫描起点/终点偏移（世界坐标，米）。
SCAN_START_OFFSET = np.array([0, 0.2, 0.1], dtype=float)
SCAN_END_OFFSET = np.array([0, -0.2, 0.1], dtype=float)
# 巡检结束后回到的参考空闲点。
HOME_XYZ = np.array([0.3, 0.0, 0.7], dtype=float)

# 抓拍前靠近扫描起点时保留的预备距离（米）。
APPROACH_DISTANCE = 0.2

# 常规直线插补段的步数、每步延时，以及悬停等待时间。
INTERPOLATION_STEPS_LINEAR = 20
DELAY_PER_STEP = 0.02
HOVER_WAIT_TIME = 1.0

# 为了让抵达拍照点更平滑，增加预进场、低速贴近和稳定等待参数。
SCAN_PRE_ENTRY_DISTANCE = 0.04
SCAN_SETTLE_ACCELERATION = 0.08
SCAN_SETTLE_VELOCITY = 0.03
SCAN_SETTLE_STEPS = 8
SCAN_SETTLE_DELAY_PER_STEP = 0.04
SCAN_STABILIZE_TIME = 0.3

# 运行显示、日志与命令输出桥接配置。
VERBOSE = True
STEP_DELAY = 0.5
SOLVER_VERBOSE = False
COMMAND_OUTPUT_ENABLED = True
COMMAND_ROBOT_HOST = None
COMMAND_ROBOT_PORT = 30002
COMMAND_SEND_STAGES = ("solver_final", "linear_step")
COMMAND_ACCELERATION = 0.35
COMMAND_VELOCITY = 0.25
COMMAND_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "robot_command_output", "task2_movej_commands.txt")

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def joint_qpos_addresses(model: mujoco.MjModel):
    return [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in JOINT_NAMES
    ]


def print_control_panel(vision_target: np.ndarray, scan_start: np.ndarray, scan_end: np.ndarray):
    orient_dir = vision_target - scan_start
    orient_dir = orient_dir / np.linalg.norm(orient_dir)

    print("\n" + "=" * 60)
    print("Task 2 control panel (Dynamic Vision Tracking)")
    print("=" * 60)
    print(f"Vision target (Center): {np.round(vision_target, 4)}")
    print(f"Scan start point   : {np.round(scan_start, 4)}")
    print(f"Scan end point     : {np.round(scan_end, 4)}")
    print(f"Home point         : {np.round(HOME_XYZ, 4)}")
    print(f"Initial joint state: {np.round(INITIAL_ANGLES, 4)}")
    print(f"Dynamic Orient Dir : {np.round(orient_dir, 4)}")
    print(f"Approach distance  : {APPROACH_DISTANCE} m")
    print(f"Pre-entry distance : {SCAN_PRE_ENTRY_DISTANCE} m")
    print(f"Settle speed       : a={SCAN_SETTLE_ACCELERATION}, v={SCAN_SETTLE_VELOCITY}")
    print(f"Stabilize wait     : {SCAN_STABILIZE_TIME} s")
    print(f"Hover time         : {HOVER_WAIT_TIME} s")
    print(f"Command output     : {COMMAND_OUTPUT_ENABLED} ({COMMAND_SEND_STAGES})")
    print(f"Robot endpoint     : {COMMAND_ROBOT_HOST}:{COMMAND_ROBOT_PORT}")
    print(f"Command file       : {COMMAND_OUTPUT_PATH}")
    print("=" * 60)


def configure_subtasks(vision_target: np.ndarray, scan_start: np.ndarray):
    direction = (vision_target - scan_start).tolist()

    if hasattr(task_orient, "TARGET_DIRECTION"):
        task_orient.TARGET_DIRECTION = direction
    if hasattr(task_approach, "TARGET_DIRECTION"):
        task_approach.TARGET_DIRECTION = direction
    if hasattr(task_reach, "TARGET_DIRECTION"):
        task_reach.TARGET_DIRECTION = direction

    if hasattr(task_approach, "APPROACH_DISTANCE"):
        task_approach.APPROACH_DISTANCE = APPROACH_DISTANCE
    if hasattr(task_reach, "GRIPPER_OFFSET"):
        task_reach.GRIPPER_OFFSET = 0.0

    if hasattr(task_arc, "ARC_CENTER"):
        task_arc.ARC_CENTER = vision_target.tolist()

    if hasattr(task_hover, "HOVER_TIME"):
        task_hover.HOVER_TIME = HOVER_WAIT_TIME
    if hasattr(task_linear, "INTERPOLATION_STEPS"):
        task_linear.INTERPOLATION_STEPS = INTERPOLATION_STEPS_LINEAR
    if hasattr(task_linear, "DELAY_PER_STEP"):
        task_linear.DELAY_PER_STEP = DELAY_PER_STEP


def _sleep_with_viewer(viewer, duration: float, dt: float = 0.02):
    steps = max(1, int(duration / dt))
    for _ in range(steps):
        if viewer:
            viewer.sync()
        time.sleep(dt)


def _get_command_client(solver):
    bridge = getattr(solver, "motion_bridge", None)
    if bridge is not None:
        return bridge
    return getattr(solver, "command_client", None)


def execute_low_speed_scan_entry(
    solver,
    model_mj,
    data_mj,
    viewer,
    current_angles,
    current_pos,
    current_rot,
    target_xyz,
):
    print(
        "[Task 2] Executing low-speed final approach "
        f"(a={SCAN_SETTLE_ACCELERATION}, v={SCAN_SETTLE_VELOCITY})..."
    )

    command_client = _get_command_client(solver)
    previous_a = getattr(command_client, "a", None) if command_client is not None else None
    previous_v = getattr(command_client, "v", None) if command_client is not None else None
    previous_steps = task_linear.INTERPOLATION_STEPS
    previous_delay = task_linear.DELAY_PER_STEP

    try:
        if command_client is not None:
            command_client.a = SCAN_SETTLE_ACCELERATION
            command_client.v = SCAN_SETTLE_VELOCITY
        task_linear.INTERPOLATION_STEPS = SCAN_SETTLE_STEPS
        task_linear.DELAY_PER_STEP = SCAN_SETTLE_DELAY_PER_STEP
        return task_linear.execute(
            solver,
            model_mj,
            data_mj,
            viewer,
            current_angles,
            current_pos,
            current_rot,
            target_xyz,
        )
    finally:
        task_linear.INTERPOLATION_STEPS = previous_steps
        task_linear.DELAY_PER_STEP = previous_delay
        if command_client is not None and previous_a is not None and previous_v is not None:
            command_client.a = previous_a
            command_client.v = previous_v


def execute_stabilize_wait(
    solver,
    model_mj,
    data_mj,
    viewer,
    current_angles,
    current_pos,
    current_rot,
    target_xyz,
):
    print(f"[Task 2] Stabilizing before focus for {SCAN_STABILIZE_TIME:.2f} s...")
    _sleep_with_viewer(viewer, SCAN_STABILIZE_TIME)
    return current_angles, current_pos, current_rot, True


def execute_return_to_initial_pose(
    solver,
    model_mj,
    data_mj,
    viewer,
    current_angles,
    current_pos,
    current_rot,
    target_joint_angles,
):
    target_angles = np.asarray(target_joint_angles, dtype=float).reshape(-1)
    if target_angles.size != len(JOINT_NAMES):
        raise ValueError(f"Expected {len(JOINT_NAMES)} joint angles, got {target_angles.size}.")

    start_angles = np.asarray(current_angles, dtype=float).reshape(-1)
    if np.allclose(start_angles, target_angles, atol=1e-6):
        print("[Task 2] Already at the initial joint pose.")
        if viewer:
            viewer.sync()
        final_pos, final_rot = get_end_effector_pose(model_mj, data_mj)
        return target_angles.copy(), final_pos, final_rot, True

    print("[Task 2] Returning to the exact initial joint pose...")
    command_client = _get_command_client(solver)
    final_stage = "solver_final"
    if command_client is not None:
        stage_enabled = getattr(command_client, "stage_enabled", None)
        if callable(stage_enabled) and not stage_enabled(final_stage):
            final_stage = "linear_step"

    for i in range(1, INTERPOLATION_STEPS_LINEAR + 1):
        t = i / INTERPOLATION_STEPS_LINEAR
        interp_angles = start_angles + t * (target_angles - start_angles)
        stage = final_stage if i == INTERPOLATION_STEPS_LINEAR else "linear_step"
        apply_motion_step(
            solver,
            model_mj,
            data_mj,
            interp_angles,
            stage=stage,
            viewer=viewer,
            delay=DELAY_PER_STEP,
        )

    final_pos, final_rot = get_end_effector_pose(model_mj, data_mj)
    return target_angles.copy(), final_pos, final_rot, True


class Status(Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3


class TreeNode:
    def __init__(self, name: str):
        self.name = name

    def tick(self, blackboard):
        raise NotImplementedError


class Sequence(TreeNode):
    def __init__(self, name: str, children):
        super().__init__(name)
        self.children = children

    def tick(self, blackboard):
        print(f"[Sequence] {self.name}")
        for child in self.children:
            if child.tick(blackboard) != Status.SUCCESS:
                return Status.FAILURE
        return Status.SUCCESS


class Fallback(TreeNode):
    def __init__(self, name: str, children):
        super().__init__(name)
        self.children = children

    def tick(self, blackboard):
        print(f"[Fallback] {self.name}")
        for child in self.children:
            status = child.tick(blackboard)
            if status != Status.FAILURE:
                return status
        return Status.FAILURE


class Condition(TreeNode):
    def __init__(self, name: str, check_func):
        super().__init__(name)
        self.check_func = check_func

    def tick(self, blackboard):
        if self.check_func(blackboard):
            print(f"  [Condition] {self.name}: success")
            return Status.SUCCESS
        print(f"  [Condition] {self.name}: failure")
        return Status.FAILURE


class Action(TreeNode):
    def __init__(self, name: str, execute_func, target_key: str = "target_xyz", post_hook=None):
        super().__init__(name)
        self.execute_func = execute_func
        self.target_key = target_key
        self.post_hook = post_hook

    def tick(self, blackboard):
        print(f"\n[Action] {self.name}")
        target_pos = getattr(blackboard, self.target_key)
        if STEP_DELAY > 0:
            time.sleep(STEP_DELAY)

        angles, pos, rot, success = self.execute_func(
            blackboard.solver,
            blackboard.model,
            blackboard.data,
            blackboard.viewer,
            blackboard.current_angles,
            blackboard.current_pos,
            blackboard.current_rot,
            target_pos,
        )
        if success:
            blackboard.current_angles = angles.copy()
            blackboard.current_pos = pos.copy()
            blackboard.current_rot = rot.copy()
            if self.post_hook:
                self.post_hook(blackboard)
            if blackboard.viewer:
                blackboard.viewer.sync()
            print(f"[Action] success: {self.name}")
            return Status.SUCCESS

        if blackboard.viewer:
            blackboard.viewer.sync()
        print(f"[Action] failure: {self.name}")
        return Status.FAILURE


class Blackboard:
    def __init__(self, model, data, solver, init_angles, vision_target: np.ndarray, scan_start: np.ndarray, scan_end: np.ndarray):
        self.model = model
        self.data = data
        self.solver = solver
        self.viewer = None
        self.initial_angles = init_angles.copy()
        self.current_angles = self.initial_angles.copy()

        for idx, val in zip(joint_qpos_addresses(self.model), self.current_angles):
            self.data.qpos[idx] = val
        mujoco.mj_forward(self.model, self.data)

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.current_pos = self.data.site_xpos[site_id].copy()
        self.current_rot = self.data.site_xmat[site_id].flatten()

        self.vision_target_xyz = vision_target.copy()
        self.scan_start_xyz = scan_start.copy()
        self.scan_end_xyz = scan_end.copy()
        self.home_xyz = HOME_XYZ.copy()

        scan_direction = self.vision_target_xyz - self.scan_start_xyz
        scan_direction_norm = np.linalg.norm(scan_direction)
        if scan_direction_norm < 1e-6:
            raise ValueError("Scan start and vision target are too close to define a scan direction.")
        self.scan_direction = scan_direction / scan_direction_norm
        self.scan_pre_entry_xyz = self.scan_start_xyz - SCAN_PRE_ENTRY_DISTANCE * self.scan_direction

        self.is_scanned = False


def build_task2_tree():
    def set_scanned(bb):
        bb.is_scanned = True

    check_scanned = Condition("scan already finished?", lambda bb: bb.is_scanned)

    approach = Action("approach scan start", task_approach.execute, target_key="scan_start_xyz")
    orient = Action("orient for scan", task_orient.execute, target_key="scan_start_xyz")
    reach_pre_entry = Action("reach scan pre-entry", task_reach.execute, target_key="scan_pre_entry_xyz")
    settle_entry = Action("low-speed settle to scan start", execute_low_speed_scan_entry, target_key="scan_start_xyz")
    stabilize = Action("stabilize before focus", execute_stabilize_wait, target_key="scan_start_xyz")
    hover_1 = Action("hover for focus", task_hover.execute, target_key="scan_start_xyz")
    arc_scan = Action("arc scan", task_arc.execute, target_key="scan_end_xyz")
    hover_2 = Action("hover for data return", task_hover.execute, target_key="scan_end_xyz", post_hook=set_scanned)

    scan_sequence = Sequence(
        "execute visual inspection sequence",
        [approach, orient, reach_pre_entry, settle_entry, stabilize, hover_1, arc_scan, hover_2],
    )
    ensure_scanned = Fallback("ensure visual inspection complete", [check_scanned, scan_sequence])
    go_home = Action("return to initial joint pose", execute_return_to_initial_pose, target_key="initial_angles")

    return Sequence("task2 visual inspection", [ensure_scanned, go_home])


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    detection = prepare_vision(model, data)

    vision_target = detection.world_position
    scan_start = vision_target + SCAN_START_OFFSET
    scan_end = vision_target + SCAN_END_OFFSET

    configure_subtasks(vision_target, scan_start)

    command_client = RobotIPClient(
        host=COMMAND_ROBOT_HOST,
        port=COMMAND_ROBOT_PORT,
        enabled=COMMAND_OUTPUT_ENABLED,
        output_path=COMMAND_OUTPUT_PATH,
        a=COMMAND_ACCELERATION,
        v=COMMAND_VELOCITY,
        send_mode=COMMAND_SEND_STAGES,
        clear_on_connect=True,
    )
    command_client.connect()

    solver = GNNIKSolver(verbose=SOLVER_VERBOSE, command_client=command_client)
    blackboard = Blackboard(model, data, solver, INITIAL_ANGLES, vision_target, scan_start, scan_end)

    if VERBOSE:
        print_control_panel(vision_target, scan_start, scan_end)

    root_tree = build_task2_tree()

    with mujoco.viewer.launch_passive(model, data) as viewer_handle:
        viewer = ViewerWithCameraInset(model, data, viewer_handle)
        blackboard.viewer = viewer
        viewer.sync()
        time.sleep(STEP_DELAY)

        print("\nStarting behavior tree...")
        final_status = root_tree.tick(blackboard)

        if final_status == Status.SUCCESS:
            print("\nTask 2 finished successfully.")
            print("Running the tree a second time to confirm the fallback logic...")
            time.sleep(1.5)
            root_tree.tick(blackboard)
        else:
            print("\nTask 2 failed.")

        time.sleep(3.0)
        viewer.close()
        command_client.close()


if __name__ == "__main__":
    main()
