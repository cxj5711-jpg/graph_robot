#!/usr/bin/env python3
"""
子任务9: 悬停等待 (Hover / Wait)
用于视觉巡检时的相机对焦、拍照或数据回传，保持当前姿态不动
"""
import time
import mujoco

# ================= 🎛️ 总控台 =================
HOVER_TIME = 2.0  # 悬停等待时间(秒)
# ============================================

def execute(solver, model_mj, data_mj, viewer, current_angles, current_pos, current_rot, target_xyz):
    print(f"[Task 9] 执行悬停等待，保持当前姿态 {HOVER_TIME} 秒...")
    
    joint_qposadr = [model_mj.jnt_qposadr[mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_JOINT, n)]
                     for n in["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]]
    
    # 按照每步 0.02秒 拆分时间，保持动画渲染平滑
    steps = int(HOVER_TIME / 0.02)
    if steps <= 0:
        steps = 1
        
    for step in range(steps):
        # 持续锁死关节当前角度
        for idx, val in zip(joint_qposadr, current_angles):
            data_mj.qpos[idx] = val
        
        mujoco.mj_forward(model_mj, data_mj)
        
        if viewer:
            viewer.sync()
            time.sleep(0.02)
            
    # 悬停不改变任何位置和姿态
    return current_angles, current_pos, current_rot, True