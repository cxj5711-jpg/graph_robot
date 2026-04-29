#!/usr/bin/env python3
"""
重构后的数据生成脚本：
1. 采用【多尺度】局部采样策略。
2. 引入【全局总控台】，可自定义关节与笛卡尔空间限位，强制剔除多解空间。
"""

import argparse
import numpy as np
import mujoco
import os

# ==============================================================================
# 🎛️ 数据生成范围总控台 (Control Panel)
# ==============================================================================

# 1. 关节角度限位 (单位: 弧度)
# 【减少多解的核心】: UR10e 的默认物理限位大多是 [-6.28, 6.28]。
# 只要你把以下某些关键关节（如 shoulder_lift, elbow）卡在某个确定的半区（例如 0~3.14），
# 就能直接消除“肘部朝上/朝下”等姿态翻转引发的多解问题，让 GNN 学到完美的一一映射。
CUSTOM_JOINT_RANGES = [
    (1.57, 4.17),  # 0: shoulder_pan_joint (底座旋转，通常可全范围)
    (-3.14, 0.0),  # 1: shoulder_lift_joint (大臂抬起，限制在负区间避免奇异)
    (0.0, 3.14),  # 2: elbow_joint (肘部，强行限制为大于0，彻底锁死"肘部朝上"单一解)
    (-3.14, 3.14),  # 3: wrist_1_joint
    (0, 3.14),  # 4: wrist_2_joint
    (-3.14, 3.14)  # 5: wrist_3_joint
]

# 2. 末端执行器笛卡尔空间位置限位 (单位: 米)
# 限制工作空间，避开靠模型太近或太低的不良区域
MIN_X = 0.3
MAX_X = float('inf')  # 如果想限制最远距离，可以改这里 (如 1.0)
MIN_Y = -float('inf')
MAX_Y = float('inf')
MIN_Z = 0.2
MAX_Z = 1.0 # 比如你想限制不要太高，可以设为 1.2
# ==============================================================================

XML_PATH = os.path.join(os.path.dirname(__file__), "universal_robots_ur10e", "ur10e.xml")
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
BODY_NAMES = ["shoulder_link", "upper_arm_link", "forearm_link",
              "wrist_1_link", "wrist_2_link", "wrist_3_link"]
SITE_NAME = "attachment_site"


def load_model(xml_path):
    if not os.path.exists(xml_path): raise FileNotFoundError(f"找不到模型文件: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


def is_valid_workspace(pos):
    """检查末端位置是否在总控台定义的工作空间内"""
    return (MIN_X <= pos[0] <= MAX_X and
            MIN_Y <= pos[1] <= MAX_Y and
            MIN_Z <= pos[2] <= MAX_Z)


def generate_paired_dataset(model, data, num_samples, output_path, max_attempts=2000000, max_step=0.5):
    joint_qposadr = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
    body_ids = [model.body(name).id for name in BODY_NAMES]

    # 将总控台的限位与 MuJoCo 物理限位取交集，确保绝对安全
    joint_ranges = []
    for i, name in enumerate(JOINT_NAMES):
        phys_min, phys_max = model.jnt_range[model.joint(name).id]
        user_min, user_max = CUSTOM_JOINT_RANGES[i]
        final_min = max(phys_min, user_min)
        final_max = min(phys_max, user_max)
        joint_ranges.append((final_min, final_max))

    dataset = np.zeros((num_samples, 96))
    i = 0
    total_attempts = 0
    print(f"开始生成【高精度空间与关节约束】数据集... 目标: {num_samples}")

    while i < num_samples and total_attempts < max_attempts:
        total_attempts += 1

        # 1. 采样起始姿态 (使用新定义的严格范围)
        cur_angles = np.array([np.random.uniform(r[0], r[1]) for r in joint_ranges])
        for j, adr in enumerate(joint_qposadr): data.qpos[adr] = cur_angles[j]
        mujoco.mj_forward(model, data)

        # 空间校验
        ee_pos_now = data.site(SITE_NAME).xpos
        if not is_valid_workspace(ee_pos_now): continue

        # 2. 多尺度步长逻辑
        rand_val = np.random.rand()
        if rand_val < 0.35:
            cur_max = max_step * 0.05  # 极小步强化 (精度)
        elif rand_val < 0.75:
            cur_max = max_step * 0.3  # 中等步
        else:
            cur_max = max_step  # 大步 (泛化)

        delta = np.random.uniform(-cur_max, cur_max, size=6)
        tgt_angles = cur_angles + delta

        # 验证目标角度是否也在限制范围内
        if any(tgt_angles < [r[0] for r in joint_ranges]) or any(tgt_angles > [r[1] for r in joint_ranges]):
            continue

        # 3. 验证目标点位
        for j, adr in enumerate(joint_qposadr): data.qpos[adr] = tgt_angles[j]
        mujoco.mj_forward(model, data)
        tgt_ee_pos = data.site(SITE_NAME).xpos.copy()
        tgt_ee_rot = data.site(SITE_NAME).xmat.copy().flatten()

        if not is_valid_workspace(tgt_ee_pos): continue

        # 4. 记录数据
        for j, adr in enumerate(joint_qposadr): data.qpos[adr] = cur_angles[j]
        mujoco.mj_forward(model, data)
        cur_poses = np.array([data.xpos[bid] for bid in body_ids]).flatten()
        cur_mats = np.array([data.xmat[bid] for bid in body_ids]).flatten()

        row = np.concatenate([cur_angles, cur_poses, cur_mats, tgt_ee_pos, tgt_ee_rot, delta])
        dataset[i] = row
        i += 1

        if i % 1000 == 0:
            print(f"进度: {i}/{num_samples} (尝试采样次数: {total_attempts})")

    if i < num_samples:
        print(f"⚠️ 警告: 达到最大尝试次数，仅生成了 {i} 个样本。请检查是否约束过严。")

    np.save(output_path, dataset[:i])
    print(f"✅ 数据生成完成，存放至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=40000)
    parser.add_argument("--output", type=str, default="ur10e_paired_data.npy")
    args = parser.parse_args()
    m, d = load_model(XML_PATH)
    generate_paired_dataset(m, d, args.num_samples, args.output)