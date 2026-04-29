#!/usr/bin/env python3
"""
子任务8: 回退 (Retreat) - 向量解耦版
严格沿 `task_orient` 中设定的三维轴向量 (TARGET_DIRECTION) 反向回退指定距离
"""
import numpy as np

# 直接引入你在姿态调整中设定的全局三维向量
TARGET_DIRECTION = [1, 1, -1]

# ================= 🎛️ 总控台 =================
RETREAT_DISTANCE = 0.2  # 回退距离 (米)
# ============================================

def execute(solver, model_mj, data_mj, viewer, current_angles, current_pos, current_rot, target_xyz):
    print(f"[Task 8] 准备安全回退，沿设定的轴向量 {TARGET_DIRECTION} 反向回退 {RETREAT_DISTANCE}m...")
    
    # 1. 获取并归一化指定的轴向量
    axis_vector = np.array(TARGET_DIRECTION, dtype=float)
    norm = np.linalg.norm(axis_vector)
    if norm < 1e-6:
        raise ValueError("TARGET_DIRECTION 不能为零向量！")
    axis_direction = axis_vector / norm
    
    # 2. 沿着该轴向量的【反方向】计算回退目标绝对坐标
    retreat_target_pos = current_pos - RETREAT_DISTANCE * axis_direction
        
    from subtask.task_linear import execute as linear_execute
    return linear_execute(solver, model_mj, data_mj, viewer, current_angles, current_pos, current_rot, retreat_target_pos)
