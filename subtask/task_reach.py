#!/usr/bin/env python3
"""
子任务7: 抵达 (Reach) - 向量统一版
严格沿 `task_orient` 中设定的三维轴向量 (TARGET_DIRECTION) 进行夹爪偏置计算
"""
import numpy as np
from subtask.task_orient import TARGET_DIRECTION

# ================= 🎛️ 总控台 =================
GRIPPER_OFFSET = 0.145  # 夹爪中心与实际抓取点的偏置距离(米)
# ============================================

def execute(solver, model_mj, data_mj, viewer, current_angles, current_pos, current_rot, target_xyz):
    # 1. 获取并归一化指定的轴向量
    axis_vector = np.array(TARGET_DIRECTION, dtype=float)
    norm = np.linalg.norm(axis_vector)
    if norm < 1e-6:
        raise ValueError("TARGET_DIRECTION 不能为零向量！")
    axis_direction = axis_vector / norm
    
    print(f"[Task 7] 准备抵达抓取点，沿向量 {TARGET_DIRECTION} 预留偏置 {GRIPPER_OFFSET}m...")
    
    # 2. 绝对坐标计算：抓取目标点 - 沿着向量反方向退回夹爪长度
    # 因为 TARGET_DIRECTION 指向物体，所以减去它就是往外退
    compensated_pos = target_xyz - GRIPPER_OFFSET * axis_direction
    
    # 执行直线插补前往计算出的绝对点
    from subtask.task_linear import execute as linear_execute
    return linear_execute(solver, model_mj, data_mj, viewer, current_angles, current_pos, current_rot, compensated_pos)