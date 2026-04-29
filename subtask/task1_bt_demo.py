#!/usr/bin/env python3
"""Behavior-tree demo for grasp and transfer using the vision target."""

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

# =========================
# Task 1 总控超参数
# =========================
# 任务开始时的 6 关节初始角（弧度）。
INITIAL_ANGLES = np.array([3.14, -1.57, 1.57, 1.57, 1.57, 0.0], dtype=float)
# 显式转移目标点（世界坐标，米）。Task1 固定采用绝对坐标方案。
PLACE_TARGET_XYZ = np.array([0.60, 0.20, 0.275], dtype=float)
# 对放置点 z 轴做下限保护，避免太贴近桌面。
PLACE_MIN_Z = 0.20

# 抓取前靠近阶段的预备距离（米）。
APPROACH_DISTANCE = 0.25
# 末端朝向向量，供 orient/approach/reach/retreat 共用。
ORIENT_DIRECTION = np.array([0, 0, -1.0], dtype=float)
# 抵达补偿（米），通常用于夹爪/工具长度补偿。
GRIPPER_OFFSET = 0.145
# 夹爪闭合/张开控制量。
GRASP_FORCE = 105
RELEASE_FORCE = 0
# 抓取后/放置后的回退距离（米）。
RETREAT_DISTANCE = 0.2
# 直线插补步数与每步延时。
INTERPOLATION_STEPS_LINEAR = 10
DELAY_PER_STEP = 0.05
PHYSICS_STEPS_PER_MOTION_STEP = 50
PHYSICS_DRIVEN_MOTION = True
PLACE_SUCCESS_TOLERANCE = 0.03
# 抓取/松开后的物理结算等待时间（秒）。
GRASP_WAIT_TIME = 3.0
RELEASE_WAIT_TIME = 3.0

# 运行显示、日志与行为树节拍控制。
VERBOSE = True
STEP_DELAY = 0.5
SOLVER_VERBOSE = False
COMMAND_OUTPUT_ENABLED = True
COMMAND_ROBOT_HOST = None
COMMAND_ROBOT_PORT = 30002
COMMAND_SEND_STAGES = ("solver_final", "linear_step")
COMMAND_ACCELERATION = 0.35
COMMAND_VELOCITY = 0.25
COMMAND_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "robot_command_output", "task1_movej_commands.txt")


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
TARGET_BODY_NAME = "target_cube"


def joint_qpos_addresses(model: mujoco.MjModel):
    return [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in JOINT_NAMES
    ]


def target_body_qpos_address(model: mujoco.MjModel, body_name: str = TARGET_BODY_NAME) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Cannot find body '{body_name}' in scene.")
    joint_adr = model.body_jntadr[body_id]
    if joint_adr < 0:
        raise ValueError(f"Body '{body_name}' does not have a free/slide/ball joint.")
    return model.jnt_qposadr[joint_adr]


def compute_place_release_xyz(place_xyz: np.ndarray) -> np.ndarray:
    axis_vector = np.asarray(ORIENT_DIRECTION, dtype=float)
    norm = np.linalg.norm(axis_vector)
    if norm < 1e-6:
        raise ValueError("ORIENT_DIRECTION cannot be a zero vector.")

    axis_direction = axis_vector / norm
    return np.asarray(place_xyz, dtype=float).copy() - GRIPPER_OFFSET * axis_direction


def print_control_panel(target_xyz: np.ndarray, place_xyz: np.ndarray):
    print("\n" + "=" * 60)
    print("Task 1 control panel")
    print("=" * 60)
    print(f"Vision target point : {np.round(target_xyz, 4)}")
    print(f"Place target xyz    : {np.round(PLACE_TARGET_XYZ, 4)}")
    print(f"Place point         : {np.round(place_xyz, 4)}")
    print(f"Place min z clamp   : {PLACE_MIN_Z} m")
    print(f"Initial joint state : {np.round(INITIAL_ANGLES, 4)}")
    print(f"Approach distance   : {APPROACH_DISTANCE} m")
    print(f"Orient direction    : {ORIENT_DIRECTION}")
    print(f"Gripper offset      : {GRIPPER_OFFSET} m")
    print(f"Grasp force         : {GRASP_FORCE}")
    print(f"Retreat distance    : {RETREAT_DISTANCE} m")
    print(f"Linear steps        : {INTERPOLATION_STEPS_LINEAR}")
    print(f"Physics steps/pose  : {PHYSICS_STEPS_PER_MOTION_STEP}")
    print(f"Physics-driven arm  : {PHYSICS_DRIVEN_MOTION}")
    print(f"Place tolerance     : {PLACE_SUCCESS_TOLERANCE} m")
    print("Transfer mode       : lift -> horizontal -> vertical")
    print(f"Step delay          : {STEP_DELAY} s")
    print(f"Command output      : {COMMAND_OUTPUT_ENABLED} ({COMMAND_SEND_STAGES})")
    print(f"Robot endpoint      : {COMMAND_ROBOT_HOST}:{COMMAND_ROBOT_PORT}")
    print(f"Command file        : {COMMAND_OUTPUT_PATH}")
    print("=" * 60)


def configure_subtasks():
    if hasattr(task_approach, "APPROACH_DISTANCE"):
        task_approach.APPROACH_DISTANCE = APPROACH_DISTANCE
    if hasattr(task_orient, "TARGET_DIRECTION"):
        task_orient.TARGET_DIRECTION = ORIENT_DIRECTION.tolist()
    if hasattr(task_approach, "TARGET_DIRECTION"):
        task_approach.TARGET_DIRECTION = ORIENT_DIRECTION.tolist()
    if hasattr(task_reach, "TARGET_DIRECTION"):
        task_reach.TARGET_DIRECTION = ORIENT_DIRECTION.tolist()
    if hasattr(task_retreat, "TARGET_DIRECTION"):
        task_retreat.TARGET_DIRECTION = ORIENT_DIRECTION.tolist()
    if hasattr(task_reach, "GRIPPER_OFFSET"):
        task_reach.GRIPPER_OFFSET = GRIPPER_OFFSET
    if hasattr(task_grasp, "GRASP_FORCE"):
        task_grasp.GRASP_FORCE = GRASP_FORCE
    if hasattr(task_grasp, "WAIT_TIME"):
        task_grasp.WAIT_TIME = GRASP_WAIT_TIME
    if hasattr(task_release, "RELEASE_FORCE"):
        task_release.RELEASE_FORCE = RELEASE_FORCE
    if hasattr(task_release, "WAIT_TIME"):
        task_release.WAIT_TIME = RELEASE_WAIT_TIME
    if hasattr(task_retreat, "RETREAT_DISTANCE"):
        task_retreat.RETREAT_DISTANCE = RETREAT_DISTANCE
    if hasattr(task_linear, "INTERPOLATION_STEPS"):
        task_linear.INTERPOLATION_STEPS = INTERPOLATION_STEPS_LINEAR
    if hasattr(task_linear, "DELAY_PER_STEP"):
        task_linear.DELAY_PER_STEP = DELAY_PER_STEP
    if hasattr(task_linear, "PHYSICS_STEPS_PER_MOTION_STEP"):
        task_linear.PHYSICS_STEPS_PER_MOTION_STEP = PHYSICS_STEPS_PER_MOTION_STEP
    if hasattr(task_linear, "PHYSICS_DRIVEN_MOTION"):
        task_linear.PHYSICS_DRIVEN_MOTION = PHYSICS_DRIVEN_MOTION


def validate_configuration() -> bool:
    return (
        getattr(task_orient, "TARGET_DIRECTION", None) == ORIENT_DIRECTION.tolist()
        and getattr(task_approach, "APPROACH_DISTANCE", None) == APPROACH_DISTANCE
        and getattr(task_reach, "GRIPPER_OFFSET", None) == GRIPPER_OFFSET
    )


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
    def __init__(self, model, data, solver, init_angles, target_xyz: np.ndarray):
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

        self.target_xyz = target_xyz.copy()
        self.place_xyz = PLACE_TARGET_XYZ.copy()
        # Vision target may be near the table. Keep place point above a safe height.
        self.place_xyz[2] = max(float(self.place_xyz[2]), PLACE_MIN_Z)
        self.place_release_xyz = compute_place_release_xyz(self.place_xyz)
        self.place_hover_xyz = self.place_release_xyz.copy()
        self.place_hover_xyz[2] = self.current_pos[2]
        self.target_qpos_adr = target_body_qpos_address(self.model, TARGET_BODY_NAME)
        self.has_object = False

    def get_target_xyz(self) -> np.ndarray:
        return self.data.qpos[self.target_qpos_adr:self.target_qpos_adr + 3].copy()


def build_task1_tree():
    def set_grasped(bb):
        bb.has_object = True
        print(f"  [Physics] grasp marked; observed object xyz: {np.round(bb.get_target_xyz(), 4)}")

    def update_place_transfer_waypoints(bb):
        # Keep the post-grasp retreat height as the transfer plane, then descend vertically.
        bb.place_release_xyz = compute_place_release_xyz(bb.place_xyz)
        bb.place_hover_xyz = bb.place_release_xyz.copy()
        bb.place_hover_xyz[2] = bb.current_pos[2]

    def set_released(bb):
        bb.has_object = False
        print(f"  [Physics] release done; observed object xyz: {np.round(bb.get_target_xyz(), 4)}")

    check_grasped = Condition("object already grasped?", lambda bb: bb.has_object)

    approach = Action("approach", task_approach.execute, target_key="target_xyz")
    orient = Action("orient", task_orient.execute, target_key="target_xyz")
    reach = Action("reach", task_reach.execute, target_key="target_xyz")
    grasp = Action("grasp", task_grasp.execute, target_key="target_xyz", post_hook=set_grasped)
    retreat_grasp = Action(
        "retreat after grasp",
        task_retreat.execute,
        target_key="target_xyz",
        post_hook=update_place_transfer_waypoints,
    )
    grasp_sequence = Sequence(
        "execute grasp sequence",
        [approach, orient, reach, grasp, retreat_grasp],
    )
    ensure_grasped = Fallback("ensure object is grasped", [check_grasped, grasp_sequence])

    move_above_place = Action("move above place point", task_linear.execute, target_key="place_hover_xyz")
    descend_to_place = Action("descend to place point", task_linear.execute, target_key="place_release_xyz")
    release = Action("release", task_release.execute, target_key="place_xyz", post_hook=set_released)
    retreat_place = Action("retreat after place", task_retreat.execute, target_key="place_xyz")
    transfer_sequence = Sequence(
        "execute transfer sequence",
        [move_above_place, descend_to_place, release, retreat_place],
    )

    return Sequence("task1 grasp and transfer", [ensure_grasped, transfer_sequence])


def main():
    print("Launching Task 1 grasp-and-transfer demo with vision target")

    configure_subtasks()
    if not validate_configuration():
        print("Subtask configuration validation failed.")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    # ================= 新增代码块 =================
    # 动态获取夹爪驱动器的 ID
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
    if actuator_id >= 0:
        # 将夹爪的最大输出力锁死在 30N，防止物理约束撕裂和发生形变
        model.actuator_forcerange[actuator_id] = [-50.0, 50.0]
    # ==============================================
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
        print_control_panel(blackboard.target_xyz, blackboard.place_xyz)

    root_tree = build_task1_tree()

    with mujoco.viewer.launch_passive(model, data) as viewer_handle:
        viewer = ViewerWithCameraInset(model, data, viewer_handle)
        blackboard.viewer = viewer
        viewer.sync()
        time.sleep(STEP_DELAY)

        print("\nStarting behavior tree...")
        final_status = root_tree.tick(blackboard)

        final_object_xyz = blackboard.get_target_xyz()
        place_err_mm = np.linalg.norm(final_object_xyz - blackboard.place_xyz) * 1000.0
        physical_success = place_err_mm <= PLACE_SUCCESS_TOLERANCE * 1000.0

        if final_status == Status.SUCCESS and physical_success:
            print("\nTask 1 finished successfully.")
        elif final_status == Status.SUCCESS:
            print("\nTask 1 motion sequence finished, but physical placement failed.")
        else:
            print("\nTask 1 failed.")

        print(f"Final object xyz    : {np.round(final_object_xyz, 4)}")
        print(f"Place target xyz    : {np.round(blackboard.place_xyz, 4)}")
        print(f"Place error         : {place_err_mm:.2f} mm")

        time.sleep(3.0)
        viewer.close()
        command_client.close()


if __name__ == "__main__":
    main()
