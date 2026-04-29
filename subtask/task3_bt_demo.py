#!/usr/bin/env python3
"""Behavior-tree demo for assembly and screw-in using the vision target."""

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
from subtask import task_grasp
from subtask import task_linear
from subtask import task_orient
from subtask import task_reach
from subtask import task_release
from subtask import task_retreat
from subtask import task_screw

# =========================
# Task 3 总控超参数
# =========================
# 任务开始时的 6 关节初始角（弧度）。
INITIAL_ANGLES = np.array([3.14, -1.57, 1.57, 1.57, 1.57, 0.0], dtype=float)
# 装配流程完成后回到的安全点。
SAFE_HOME_XYZ = np.array([0.3, 0.0, 0.6], dtype=float)
# 最终装配点：vision_target + ASSEMBLY_OFFSET_FROM_VISION。
ASSEMBLY_OFFSET_FROM_VISION = np.array([0.0, 0.0, 0.0], dtype=float)

# 末端朝向与靠近/抵达/回退的几何参数。
ORIENT_DIRECTION = np.array([0.0, 0.0, -1.0], dtype=float)
APPROACH_DISTANCE = 0.20
REACH_OFFSET = 0.145
RETREAT_DISTANCE = 0.15

# 夹爪闭合/张开控制量与松开后结算时间。
GRASP_FORCE = 170
RELEASE_FORCE = 0
RELEASE_WAIT_TIME = 1.5
# 抓取后额外保持时长，保证旋进阶段从稳定状态开始。
EXTRA_GRASP_HOLD = 0.2

# 旋进轨迹参数与求解器约束。
SCREW_TURNS = 1.0
SCREW_DEPTH = 0.03
SCREW_STEPS = 40
SCREW_STEP_DELAY = 0.05
SCREW_FEED_WITH_TARGET_DIRECTION = True
SCREW_SOLVER_MAX_ITER = 60
SCREW_SOLVER_TOL_POS = 0.008
SCREW_SOLVER_TOL_ROT = 0.01
SCREW_MAX_FAIL_RATIO = 0.10
SCREW_FINAL_POS_TOL = 0.012
SCREW_FINAL_ROT_MSE_TOL = 0.02

# 直线插补步数与每步延时。
INTERPOLATION_STEPS_LINEAR = 24
DELAY_PER_STEP = 0.05

# 运行显示、日志与行为树节拍控制。
VERBOSE = True
STEP_DELAY = 0.4
SOLVER_VERBOSE = False
COMMAND_OUTPUT_ENABLED = True
COMMAND_ROBOT_HOST = None
COMMAND_ROBOT_PORT = 30002
COMMAND_SEND_STAGES = ("solver_final", "linear_step")
COMMAND_ACCELERATION = 0.35
COMMAND_VELOCITY = 0.25
COMMAND_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "robot_command_output", "task3_movej_commands.txt")

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


def calc_grasp_wait_time() -> float:
    return SCREW_STEPS * SCREW_STEP_DELAY + EXTRA_GRASP_HOLD


def print_control_panel(vision_target: np.ndarray, assembly_target: np.ndarray):
    print("\n" + "=" * 60)
    print("Task 3 control panel")
    print("=" * 60)
    print(f"Vision target         : {np.round(vision_target, 4)}")
    print(f"Assembly target       : {np.round(assembly_target, 4)}")
    print(f"Assembly target offset: {np.round(ASSEMBLY_OFFSET_FROM_VISION, 4)}")
    print(f"Safe home point       : {np.round(SAFE_HOME_XYZ, 4)}")
    print(f"Orient direction      : {ORIENT_DIRECTION}")
    print(f"Approach distance     : {APPROACH_DISTANCE} m")
    print(f"Reach offset          : {REACH_OFFSET} m")
    print(f"Retreat distance      : {RETREAT_DISTANCE} m")
    print(f"Grasp force           : {GRASP_FORCE}")
    print(f"Screw turns           : {SCREW_TURNS}")
    print(f"Screw depth           : {SCREW_DEPTH} m")
    print(f"Screw steps           : {SCREW_STEPS}")
    print(f"Screw step delay      : {SCREW_STEP_DELAY} s")
    print(f"Grasp wait time       : {calc_grasp_wait_time():.2f} s")
    print(f"Command output        : {COMMAND_OUTPUT_ENABLED} ({COMMAND_SEND_STAGES})")
    print(f"Robot endpoint        : {COMMAND_ROBOT_HOST}:{COMMAND_ROBOT_PORT}")
    print(f"Command file          : {COMMAND_OUTPUT_PATH}")
    print("=" * 60)


def configure_subtasks():
    direction = ORIENT_DIRECTION.tolist()

    if hasattr(task_orient, "TARGET_DIRECTION"):
        task_orient.TARGET_DIRECTION = direction
    if hasattr(task_approach, "TARGET_DIRECTION"):
        task_approach.TARGET_DIRECTION = direction
    if hasattr(task_reach, "TARGET_DIRECTION"):
        task_reach.TARGET_DIRECTION = direction
    if hasattr(task_screw, "TARGET_DIRECTION"):
        task_screw.TARGET_DIRECTION = direction
    if hasattr(task_retreat, "TARGET_DIRECTION"):
        task_retreat.TARGET_DIRECTION = direction

    if hasattr(task_approach, "APPROACH_DISTANCE"):
        task_approach.APPROACH_DISTANCE = APPROACH_DISTANCE
    if hasattr(task_reach, "GRIPPER_OFFSET"):
        task_reach.GRIPPER_OFFSET = REACH_OFFSET
    if hasattr(task_retreat, "RETREAT_DISTANCE"):
        task_retreat.RETREAT_DISTANCE = RETREAT_DISTANCE

    if hasattr(task_grasp, "GRASP_FORCE"):
        task_grasp.GRASP_FORCE = GRASP_FORCE
    if hasattr(task_grasp, "WAIT_TIME"):
        task_grasp.WAIT_TIME = calc_grasp_wait_time()

    if hasattr(task_screw, "SCREW_TURNS"):
        task_screw.SCREW_TURNS = SCREW_TURNS
    if hasattr(task_screw, "SCREW_DEPTH"):
        task_screw.SCREW_DEPTH = SCREW_DEPTH
    if hasattr(task_screw, "SCREW_STEPS"):
        task_screw.SCREW_STEPS = SCREW_STEPS
    if hasattr(task_screw, "DELAY_PER_STEP"):
        task_screw.DELAY_PER_STEP = SCREW_STEP_DELAY
    if hasattr(task_screw, "SCREW_FEED_WITH_TARGET_DIRECTION"):
        task_screw.SCREW_FEED_WITH_TARGET_DIRECTION = SCREW_FEED_WITH_TARGET_DIRECTION
    if hasattr(task_screw, "SOLVER_MAX_ITER"):
        task_screw.SOLVER_MAX_ITER = SCREW_SOLVER_MAX_ITER
    if hasattr(task_screw, "SOLVER_TOL_POS"):
        task_screw.SOLVER_TOL_POS = SCREW_SOLVER_TOL_POS
    if hasattr(task_screw, "SOLVER_TOL_ROT"):
        task_screw.SOLVER_TOL_ROT = SCREW_SOLVER_TOL_ROT
    if hasattr(task_screw, "MAX_FAIL_RATIO"):
        task_screw.MAX_FAIL_RATIO = SCREW_MAX_FAIL_RATIO
    if hasattr(task_screw, "FINAL_POS_TOL"):
        task_screw.FINAL_POS_TOL = SCREW_FINAL_POS_TOL
    if hasattr(task_screw, "FINAL_ROT_MSE_TOL"):
        task_screw.FINAL_ROT_MSE_TOL = SCREW_FINAL_ROT_MSE_TOL

    if hasattr(task_release, "RELEASE_FORCE"):
        task_release.RELEASE_FORCE = RELEASE_FORCE
    if hasattr(task_release, "WAIT_TIME"):
        task_release.WAIT_TIME = RELEASE_WAIT_TIME

    if hasattr(task_linear, "INTERPOLATION_STEPS"):
        task_linear.INTERPOLATION_STEPS = INTERPOLATION_STEPS_LINEAR
    if hasattr(task_linear, "DELAY_PER_STEP"):
        task_linear.DELAY_PER_STEP = DELAY_PER_STEP


def validate_configuration() -> bool:
    issues = []
    if getattr(task_orient, "TARGET_DIRECTION", None) != ORIENT_DIRECTION.tolist():
        issues.append("task_orient.TARGET_DIRECTION")
    if getattr(task_approach, "APPROACH_DISTANCE", None) != APPROACH_DISTANCE:
        issues.append("task_approach.APPROACH_DISTANCE")
    if getattr(task_screw, "SCREW_TURNS", None) != SCREW_TURNS:
        issues.append("task_screw.SCREW_TURNS")
    if getattr(task_screw, "SCREW_DEPTH", None) != SCREW_DEPTH:
        issues.append("task_screw.SCREW_DEPTH")
    if getattr(task_retreat, "TARGET_DIRECTION", None) != ORIENT_DIRECTION.tolist():
        issues.append("task_retreat.TARGET_DIRECTION")

    if issues:
        print("Configuration validation failed:")
        for item in issues:
            print(f" - {item}")
        return False
    return True


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
            status = child.tick(blackboard)
            if status != Status.SUCCESS:
                return status
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

        final_angles, final_pos, final_rot, success = self.execute_func(
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
            blackboard.current_angles = final_angles.copy()
            blackboard.current_pos = final_pos.copy()
            blackboard.current_rot = final_rot.copy()
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
    def __init__(self, model, data, solver, init_angles: np.ndarray, vision_target: np.ndarray):
        self.model = model
        self.data = data
        self.solver = solver
        self.viewer = None
        self.current_angles = init_angles.copy()

        for idx, val in zip(joint_qpos_addresses(self.model), self.current_angles):
            self.data.qpos[idx] = val
        mujoco.mj_forward(self.model, self.data)

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.current_pos = self.data.site_xpos[site_id].copy()
        self.current_rot = self.data.site_xmat[site_id].flatten()

        self.vision_target_xyz = vision_target.copy()
        self.target_xyz = vision_target + ASSEMBLY_OFFSET_FROM_VISION
        self.safe_home_xyz = SAFE_HOME_XYZ.copy()
        self.is_assembled = False


def build_task3_tree():
    def set_assembled(bb):
        bb.is_assembled = True

    check_assembled = Condition("assembly already done?", lambda bb: bb.is_assembled)

    approach = Action("approach assembly point", task_approach.execute, target_key="target_xyz")
    orient = Action("orient for assembly", task_orient.execute, target_key="target_xyz")
    reach = Action("reach assembly point", task_reach.execute, target_key="target_xyz")
    grasp = Action("grasp preload", task_grasp.execute, target_key="target_xyz")
    screw = Action("screw-in assembly", task_screw.execute, target_key="target_xyz")
    release = Action("release gripper", task_release.execute, target_key="target_xyz", post_hook=set_assembled)

    assembly_sequence = Sequence(
        "execute assembly sequence",
        [approach, orient, reach, grasp, screw, release],
    )
    ensure_assembled = Fallback("ensure assembly complete", [check_assembled, assembly_sequence])

    retreat = Action("retreat after assembly", task_retreat.execute, target_key="target_xyz")
    go_safe = Action("move to safe home", task_linear.execute, target_key="safe_home_xyz")
    return Sequence("task3 assembly and screw-in", [ensure_assembled, retreat, go_safe])


def main():
    print("Launching Task 3 assembly-and-screw demo with vision target")

    configure_subtasks()
    if not validate_configuration():
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    detection = prepare_vision(model, data)

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
    blackboard = Blackboard(model, data, solver, INITIAL_ANGLES, detection.world_position)

    if VERBOSE:
        print_control_panel(blackboard.vision_target_xyz, blackboard.target_xyz)

    root_tree = build_task3_tree()

    with mujoco.viewer.launch_passive(model, data) as viewer_handle:
        viewer = ViewerWithCameraInset(model, data, viewer_handle)
        blackboard.viewer = viewer
        viewer.sync()
        time.sleep(STEP_DELAY)

        print("\nStarting behavior tree...")
        final_status = root_tree.tick(blackboard)

        if final_status == Status.SUCCESS:
            print("\nTask 3 finished successfully.")
        else:
            print("\nTask 3 failed.")

        time.sleep(3.0)
        viewer.close()
        command_client.close()


if __name__ == "__main__":
    main()
