#!/usr/bin/env python3
"""
重构后的 IK 求解器：
1. 完美对齐生成数据的【全局总控台】约束（x,y,z 及 各关节限位）。
2. 在受限的“一一映射”空间内，进行全局随机起点/终点采样，测试模型的全局泛化能力。
3. 实时输出加权损失(Joint, Pos, Rot)监控，日志格式不变。
"""

import numpy as np
import torch
import mujoco
import mujoco.viewer
import time
import argparse
from GNN.dataset import IKDataset
from model import IKGATModel
from torch_geometric.data import Data

# ==============================================================================
# 🎛️ 求解器范围总控台 (必须与 generate_ur10e_data.py 保持完全一致)
# ==============================================================================

# 1. 关节角度限位 (消除多解的核心)
CUSTOM_JOINT_RANGES = [
    (1.57, 4.17),  # 0: shoulder_pan_joint
    (-3.14, 0.0),  # 1: shoulder_lift_joint
    (0.0, 3.14),  # 2: elbow_joint
    (-3.14, 3.14),  # 3: wrist_1_joint
    (0, 3.14),  # 4: wrist_2_joint
    (-3.14, 3.14)  # 5: wrist_3_joint
]

# 2. 末端执行器笛卡尔空间位置限位
MIN_X = 0.3
MAX_X = float('inf')
MIN_Y = -float('inf')
MAX_Y = float('inf')
MIN_Z = 0.2
MAX_Z = 1.0
# ==============================================================================

# ========== 用户配置区域 ==========
XML_PATH = "../universal_robots_ur10e/ur10e.xml"
MAX_ITER = 50
TOL_POS = 0.002
TOL_ANGLE = 0.01
STEP_CLIP = 0.5  # 严禁单步预测超过0.5rad
DELAY = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SITE_NAME = "attachment_site"
# ====================================

np.set_printoptions(suppress=True, precision=8)


def random_joint_angles(joint_ranges):
    return np.array([np.random.uniform(low, high) for low, high in joint_ranges])


def is_end_effector_valid(model_mj, data_mj, joint_qposadr, angles):
    """严格按照总控台的笛卡尔空间限制验证"""
    for idx, val in zip(joint_qposadr, angles):
        data_mj.qpos[idx] = val
    mujoco.mj_forward(model_mj, data_mj)
    pos = data_mj.site(SITE_NAME).xpos
    return (MIN_X <= pos[0] <= MAX_X and
            MIN_Y <= pos[1] <= MAX_Y and
            MIN_Z <= pos[2] <= MAX_Z)


def get_end_effector_pose(model_mj, data_mj, joint_qposadr, angles):
    for idx, val in zip(joint_qposadr, angles):
        data_mj.qpos[idx] = val
    mujoco.mj_forward(model_mj, data_mj)
    pos = data_mj.site(SITE_NAME).xpos.copy()
    rot = data_mj.site(SITE_NAME).xmat.copy().flatten()
    return pos, rot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="best_gat_model.pt")
    parser.add_argument('--data_path', type=str, default="ur10e_paired_data.npy")
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    args = parser.parse_args()

    device = torch.device(DEVICE)

    # ---------- 加载 MuJoCo 模型 ----------
    model_mj = mujoco.MjModel.from_xml_path(XML_PATH)
    data_mj = mujoco.MjData(model_mj)

    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    joint_qposadr = [model_mj.jnt_qposadr[mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in
                     joint_names]

    # 将总控台的限位与 MuJoCo 物理限位取交集，计算最终合法的关节范围
    joint_ranges = []
    for i, n in enumerate(joint_names):
        phys_min, phys_max = model_mj.jnt_range[mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_JOINT, n)]
        user_min, user_max = CUSTOM_JOINT_RANGES[i]
        final_min = max(phys_min, user_min)
        final_max = min(phys_max, user_max)
        joint_ranges.append((final_min, final_max))

    body_ids = [mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, n) for n in
                ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]]
    site_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)

    # ---------- 加载标准化参数 ----------
    dataset = IKDataset(data_file=args.data_path, normalize=True)
    mean_np = dataset.mean_np
    std_np = dataset.std_np
    edge_index = dataset.edge_index.to(device)

    # ---------- 加载模型 ----------
    # 🔥 修改：in_channels 从 31 改为 32
    model = IKGATModel(in_channels=32, hidden_channels=args.hidden, num_layers=args.layers, heads=args.heads).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 🔥 核心修改：在约束空间内进行“全局随机独立采样”，不再依赖 delta 增量 🔥
    print("\n寻找符合空间与关节约束的全局随机测试姿态...")
    start_angles, target_angles = None, None
    for _ in range(50000):
        # 完全独立地采样起点和终点
        s = random_joint_angles(joint_ranges)
        t = random_joint_angles(joint_ranges)

        # 验证它们是否都在合法的笛卡尔空间内
        if is_end_effector_valid(model_mj, data_mj, joint_qposadr, s) and \
                is_end_effector_valid(model_mj, data_mj, joint_qposadr, t):
            start_angles, target_angles = s, t
            break

    if start_angles is None: raise RuntimeError("未能找到有效姿态，请检查空间约束是否过严")

    target_pos, target_rot = get_end_effector_pose(model_mj, data_mj, joint_qposadr, target_angles)
    current_angles = start_angles.copy()

    # 用于计算关节损失的权重矩阵 (对齐 train.py)
    joint_weights = np.array([1.2, 1.2, 1.2, 1.0, 1.0, 1.0])

    # ---------- 迭代求解与可视化 ----------
    iter_count = 0

    with mujoco.viewer.launch_passive(model_mj, data_mj) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 90, -20, 2.0

        while viewer.is_running() and iter_count < MAX_ITER:
            for idx, val in zip(joint_qposadr, current_angles): data_mj.qpos[idx] = val
            mujoco.mj_forward(model_mj, data_mj)

            cur_pos_all = np.array([data_mj.xpos[bid].copy() for bid in body_ids])
            cur_rot_all = np.array([data_mj.xmat[bid].flatten() for bid in body_ids])

            node_feats = []
            for i in range(6):
                onehot = np.zeros(6, dtype=np.float32)
                onehot[i] = 1.0
                rel_pos = target_pos - cur_pos_all[i]
                rel_rot = target_rot - cur_rot_all[i]
                # 🔥 修改：构造测试时的特征，将单标量角度改为 sin 和 cos
                raw_feat = np.concatenate([[np.sin(current_angles[i]), np.cos(current_angles[i])], cur_pos_all[i], cur_rot_all[i], rel_pos, rel_rot])
                norm_feat = (raw_feat - mean_np) / std_np
                node_feats.append(np.concatenate([onehot, norm_feat]))

            x = torch.from_numpy(np.array(node_feats)).float().to(device)

            with torch.no_grad():
                data_graph = Data(x=x, edge_index=edge_index)
                delta_pred = model(data_graph).cpu().numpy().flatten()

            # 安全锁
            delta_pred = np.clip(delta_pred, -STEP_CLIP, STEP_CLIP)
            current_angles = current_angles + delta_pred

            # 🔥 核心修改：无缝穿墙逻辑 🔥
            for idx in range(6):
                min_val, max_val = CUSTOM_JOINT_RANGES[idx]
                # 对于完整的圆形关节范围，使用周期性折叠让其穿墙
                if min_val <= -3.14 and max_val >= 3.14:
                    current_angles[idx] = (current_angles[idx] + np.pi) % (2 * np.pi) - np.pi
                else:
                    # 对于半圆限制的关节，依然使用撞墙拦截
                    current_angles[idx] = np.clip(current_angles[idx], min_val, max_val)

            # 计算当前误差
            for idx, val in zip(joint_qposadr, current_angles): data_mj.qpos[idx] = val
            mujoco.mj_forward(model_mj, data_mj)

            curr_ee_pos = data_mj.site_xpos[site_id].copy()
            curr_ee_rot = data_mj.site_xmat[site_id].copy().flatten()

            # 物理直观误差
            pos_err = np.linalg.norm(curr_ee_pos - target_pos)
            rot_err = np.mean(np.square(curr_ee_rot - target_rot))
            angle_mae = np.mean(np.abs(current_angles - target_angles))

            # 网络视角的各项 Loss 计算
            l_joint = np.mean(joint_weights * np.abs(current_angles - target_angles))
            l_pos = np.mean(np.abs(curr_ee_pos - target_pos)) * 10.0
            l_rot = np.mean(np.abs(curr_ee_rot - target_rot)) * 15.0

            print(f"迭代 {iter_count + 1:02d}: 位置误差={pos_err * 100:.4f}cm, 角度误差MAE={angle_mae:.4f}rad | "
                  f"Loss(Joint: {l_joint:.4f}, Pos: {l_pos:.4f}, Rot: {l_rot:.4f})")

            viewer.sync()
            time.sleep(DELAY)

            # 收敛条件：位置相差小于 5mm，且姿态基本对齐
            if pos_err < TOL_POS and rot_err < 0.005:
                break
            iter_count += 1

        print("\n=== 最终结果 ===")
        print(f"起始关节角度: {start_angles}")
        print(f"目标关节角度: {target_angles}")
        print(f"最终关节角度: {current_angles}")
        print(f"角度差值 (平均值): {angle_mae:.6f} rad")
        print(f"角度各关节差值: {current_angles - target_angles}")
        print(" ")
        print(f"最终末端位置: x={curr_ee_pos[0]:.6f}, y={curr_ee_pos[1]:.6f}, z={curr_ee_pos[2]:.6f}")
        print(f"目标末端位置: x={target_pos[0]:.6f}, y={target_pos[1]:.6f}, z={target_pos[2]:.6f}")

        pos_diff = curr_ee_pos - target_pos
        print(f"位置差值 (Δx, Δy, Δz)mm: {pos_diff[0]*1000:.6f}, {pos_diff[1]*1000:.6f}, {pos_diff[2]*1000:.6f}")

        while viewer.is_running(): time.sleep(0.1)


if __name__ == "__main__":
    main()