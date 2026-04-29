#!/usr/bin/env python3
"""
GNN-IK 求解器封装模块 (生产环境版)
提供类似 IKpy 的接口，加入收敛状态返回、静默开关以及极速加载缓存机制。
"""

import os
import json
import numpy as np
import torch
import mujoco

# 适配跨目录调用时的相对导包
from GNN.model import IKGATModel
from torch_geometric.data import Data

try:
    from GNN.dataset import IKDataset
except ImportError:
    pass

# ==============================================================================
# 🎛️ 全局总控台 (Control Panel) - 所有控制参数和路径配置都在这里
# ==============================================================================

# 1. 求解器行为控制
VERBOSE = False  # 终端静默开关：设为 True 打印详细误差，False 保持绝对安静
TOL_POS = 0.005  # 位置收敛容差 (米) -> 默认 5mm
TOL_ROT = 0.005  # 姿态收敛容差 (旋转矩阵 MSE)
MAX_ITER = 50  # 最大迭代推断次数
STEP_CLIP = 0.2  # 单步最大关节角度增量 (弧度)，防止网络步子迈太大导致震荡


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "best_gat_model.pt")
DEFAULT_XML_PATH = os.path.join(ROOT_DIR, "universal_robots_ur10e", "ur10e.xml")
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "ur10e_paired_data.npy")
DEFAULT_STATS_PATH = os.path.join(BASE_DIR, "ik_stats.json")  # 数据集均值/方差极速缓存文件

# 3. 关节限位配置（强制防止奇点与多解，必须与训练时一致）
CUSTOM_JOINT_RANGES = [
    (1.57, 4.17),  # shoulder_pan_joint
    (-3.14, 0.0),  # shoulder_lift_joint
    (0.0, 3.14),  # elbow_joint
    (-3.14, 3.14),  # wrist_1_joint
    (0, 3.14),  # wrist_2_joint
    (-3.14, 3.14)  # wrist_3_joint
]


# ==============================================================================

class GNNIKSolver:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, xml_path=DEFAULT_XML_PATH,
                 data_path=DEFAULT_DATA_PATH, device='cuda', verbose=VERBOSE,
                 command_client=None, motion_bridge=None):
        """
        初始化求解器，加载模型和 MuJoCo 引擎。使用缓存机制极速加载标准化参数。
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.motion_bridge = motion_bridge if motion_bridge is not None else command_client
        self.command_client = self.motion_bridge
        if verbose:
            print(f"[IK Solver] 初始化开始，运行设备: {self.device}")

        # ---------- 1. 加载或计算标准化参数 (极速缓存机制) ----------
        if os.path.exists(DEFAULT_STATS_PATH):
            if verbose: print(f"[IK Solver] 命中缓存，极速加载标准化参数: {DEFAULT_STATS_PATH}")
            with open(DEFAULT_STATS_PATH, 'r') as f:
                stats = json.load(f)
            self.mean_np = np.array(stats['mean'], dtype=np.float32)
            self.std_np = np.array(stats['std'], dtype=np.float32)
        else:
            if verbose: print(f"[IK Solver] 未找到缓存，正在加载完整数据集计算并生成 {DEFAULT_STATS_PATH} ...")
            dataset = IKDataset(data_file=data_path, normalize=True)
            self.mean_np = dataset.mean_np
            self.std_np = dataset.std_np
            with open(DEFAULT_STATS_PATH, 'w') as f:
                json.dump({'mean': self.mean_np.tolist(), 'std': self.std_np.tolist()}, f)

        # 构建 edge_index (无需依赖 Dataset 即可生成 6 关节连边)
        src = list(range(5))
        dst = list(range(1, 6))
        self.edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long).to(self.device)

        # ---------- 2. 加载 GAT 模型 ----------
        if verbose: print(f"[IK Solver] 加载模型权重: {model_path}")
        self.model = IKGATModel(in_channels=32, hidden_channels=256, num_layers=4, heads=4).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # ---------- 3. 加载 MuJoCo 模型 ----------
        if verbose: print(f"[IK Solver] 加载 MuJoCo 模型: {xml_path}")
        self.model_mj = mujoco.MjModel.from_xml_path(xml_path)
        self.data_mj = mujoco.MjData(self.model_mj)

        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_qposadr = [self.model_mj.jnt_qposadr[mujoco.mj_name2id(self.model_mj, mujoco.mjtObj.mjOBJ_JOINT, n)]
                              for n in joint_names]

        self.joint_ranges = CUSTOM_JOINT_RANGES

        body_names = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
        self.body_ids = [mujoco.mj_name2id(self.model_mj, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names]
        self.site_id = mujoco.mj_name2id(self.model_mj, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

        if verbose: print("[IK Solver] 初始化完成已就绪！\n")

    def on_joint_state_applied(self, joint_angles, stage):
        return self.emit_command(joint_angles, stage)

    def emit_command(self, joint_angles, stage):
        bridge = self.motion_bridge
        if bridge is None:
            return None
        bridge_emit = getattr(bridge, "emit_joint_command", None)
        if callable(bridge_emit):
            return bridge_emit(joint_angles, stage=stage)
        stage_enabled = getattr(bridge, "stage_enabled", None)
        if callable(stage_enabled) and not stage_enabled(stage):
            return None
        send_movej = getattr(bridge, "send_movej", None)
        if callable(send_movej):
            return send_movej(joint_angles)
        return None

    def solve(self, current_angles, target_pos, target_rot,
              max_iter=MAX_ITER, tol_pos=TOL_POS, tol_rot=TOL_ROT, step_clip=STEP_CLIP, verbose=VERBOSE,
              emit_solver_iteration=True, emit_solver_final=True):
        """
        核心求解接口。
        :return: (angles, success) - 返回 最终关节角度数组(6,) 和 是否完全收敛的布尔值
        """
        angles = current_angles.copy().astype(np.float64)
        success = False  # 默认未收敛

        for iter_count in range(max_iter):
            # 1. 更新 MuJoCo 正运动学
            for idx, val in zip(self.joint_qposadr, angles):
                self.data_mj.qpos[idx] = val
            mujoco.mj_forward(self.model_mj, self.data_mj)

            cur_pos_all = np.array([self.data_mj.xpos[bid].copy() for bid in self.body_ids])
            cur_rot_all = np.array([self.data_mj.xmat[bid].flatten() for bid in self.body_ids])

            # 2. 构建特征矩阵
            node_feats = []
            for i in range(6):
                onehot = np.zeros(6, dtype=np.float32)
                onehot[i] = 1.0
                rel_pos = target_pos - cur_pos_all[i]
                rel_rot = target_rot - cur_rot_all[i]

                raw_feat = np.concatenate(
                    [[np.sin(angles[i]), np.cos(angles[i])], cur_pos_all[i], cur_rot_all[i], rel_pos, rel_rot])
                norm_feat = (raw_feat - self.mean_np) / self.std_np
                node_feats.append(np.concatenate([onehot, norm_feat]).tolist())

            # Avoid torch<->numpy direct bridge for environments where
            # torch was built against NumPy 1.x but runtime uses NumPy 2.x.
            x = torch.tensor(node_feats, dtype=torch.float32, device=self.device)

            # 3. 网络预测增量
            with torch.no_grad():
                delta_tensor = self.model(Data(x=x, edge_index=self.edge_index)).view(-1)
                delta = np.asarray(delta_tensor.detach().cpu().tolist(), dtype=np.float64)

            # 4. 步长裁剪与角度更新
            delta = np.clip(delta, -step_clip, step_clip)
            angles = angles + delta

            # 5. 防穿墙限位处理
            for i in range(6):
                min_val, max_val = self.joint_ranges[i]
                if min_val <= -np.pi and max_val >= np.pi:
                    angles[i] = (angles[i] + np.pi) % (2 * np.pi) - np.pi
                else:
                    angles[i] = np.clip(angles[i], min_val, max_val)

            # 6. 验证收敛性
            for idx, val in zip(self.joint_qposadr, angles):
                self.data_mj.qpos[idx] = val
            mujoco.mj_forward(self.model_mj, self.data_mj)

            if emit_solver_iteration:
                self.emit_command(angles, stage="solver_iteration")
            curr_pos = self.data_mj.site_xpos[self.site_id].copy()
            curr_rot = self.data_mj.site_xmat[self.site_id].flatten()

            pos_err = np.linalg.norm(curr_pos - target_pos)
            rot_err = np.mean(np.square(curr_rot - target_rot))

            # 收敛打断
            if pos_err < tol_pos and rot_err < tol_rot:
                success = True
                break

        # 仅在 verbose=True 时打印详细日志，保持默认状态的绝对静默
        if verbose:
            print("\n=== IK 求解结果 ===")
            print(f"收敛状态: {'✅ 成功' if success else '❌ 未完全收敛'}")
            print(f"迭代次数: {iter_count + 1} / {max_iter}")
            print(f"最终末端位置: x={curr_pos[0]:.6f}, y={curr_pos[1]:.6f}, z={curr_pos[2]:.6f}")
            print(f"目标末端位置: x={target_pos[0]:.6f}, y={target_pos[1]:.6f}, z={target_pos[2]:.6f}")
            print(
                f"位置差值 Δx, Δy, Δz (mm): {(curr_pos[0] - target_pos[0]) * 1000:.2f}, {(curr_pos[1] - target_pos[1]) * 1000:.2f}, {(curr_pos[2] - target_pos[2]) * 1000:.2f}")
            print(f"姿态误差 MSE: {rot_err:.6f}")

        if emit_solver_final and success:
            self.emit_command(angles, stage="solver_final")
        return angles, success
