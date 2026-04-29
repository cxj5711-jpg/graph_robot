#!/usr/bin/env python3
"""
子任务6: 夹爪松开 (Release)
"""
import time
import mujoco

from subtask.runtime_bridge import apply_joint_controls

# ================= 🎛️ 总控台 =================
RELEASE_FORCE = 0   # 夹爪松开力度 (0表示完全张开)
WAIT_TIME = 4.0     # 物理引擎停顿结算时间(秒)
SIM_STEP = 0.02
# ============================================

def execute(solver, model_mj, data_mj, viewer, current_angles, current_pos, current_rot, target_xyz):
    print(f"[Task 6] 执行夹爪松开，物理结算等待 {WAIT_TIME}秒...")
    gripper_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
    
    joint_qposadr = [model_mj.jnt_qposadr[mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_JOINT, n)]
                     for n in ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]]
    
    release_steps = max(1, int(round(WAIT_TIME / SIM_STEP)))
    
    for step in range(release_steps):
        # 锁死手臂当前姿态，防止受到物理重力掉落
        for idx, val in zip(joint_qposadr, current_angles):
            data_mj.qpos[idx] = val
        apply_joint_controls(model_mj, data_mj, current_angles)
            
        data_mj.ctrl[gripper_id] = RELEASE_FORCE
        mujoco.mj_step(model_mj, data_mj)
        
        if viewer:
            viewer.sync()
            time.sleep(SIM_STEP)
            
    return current_angles, current_pos, current_rot, True
