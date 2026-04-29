import socket
import math

ROBOT_IP = "192.168.125.12"
PORT = 30002

# 从上到下依次为 j0, j1, j2, j3, j4, j5
joints = [
    math.radians(-6.17),    # j0 底座（不动）
    math.radians(-66.67),   # j1 肩部（不动）
    math.radians(117.03),    # j2 肘部（可动，但你没要求动，所以保持）
    math.radians(-105.95),  # j3 手部1（保持）
    math.radians(-95.60),   # j4 手部2（保持）
    math.radians(195.29) + 0.01  # j5 手部3（增加0.01弧度，约0.57°）
]

# 构造命令
joint_str = ",".join([f"{j:.6f}" for j in joints])
cmd = f"movej([{joint_str}], a=0.25, v=0.55)\n"
print("发送命令:", cmd.strip())

# 发送
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ROBOT_IP, PORT))
    sock.send(cmd.encode())
    print("命令发送成功！")
    sock.close()
except Exception as e:
    print("发送失败:", e)