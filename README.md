# Graph Robot

面向 UR10e 机械臂的图神经网络逆运动学求解与多任务运动规划项目。项目结合 GATv2 逆运动学模型、MuJoCo 仿真场景、视觉目标估计与行为树式任务脚本，用于验证机械臂在抓取转移、视觉巡检和旋进装配等复合任务中的规划流程。

## 项目特点

- 基于 GATv2 的六自由度 UR10e 逆运动学增量预测模型。
- 使用 MuJoCo 对机械臂模型、末端位姿和任务场景进行仿真验证。
- 提供视觉目标估计流程，可从仿真相机图像和深度信息恢复目标三维位置。
- 将复杂任务拆分为 approach、orient、reach、linear、grasp、release、retreat、screw 等可复用子任务。
- 默认输出 URScript 风格的 `movej(...)` 文本指令，便于后续迁移到真实机器人控制链路。

## 目录结构

```text
.
├── GNN/                         # GNN 逆运动学模型、数据生成、训练与求解接口
├── subtask/                     # 任务原语与三个复合任务行为树示例
├── universal_robots_ur10e/      # UR10e MuJoCo 模型、场景与网格资源
├── robot_command_output/        # 任务运行后生成的 movej 指令文本
├── docs/                        # 实验规划与真实机器人 IP 控制说明文档
├── run_vision.py                # MuJoCo 视觉目标估计与演示入口
├── robot_ip_client.py           # 机器人指令输出与可选 TCP 发送适配器
├── gripper_client.py            # 夹爪控制客户端
└── real_robot_ip_migration_guide.md
```

## 环境依赖

建议使用 Python 3.10 或更高版本，并根据本机 CUDA / CPU 环境安装 PyTorch 与 PyTorch Geometric。

核心依赖包括：

```text
numpy
torch
torch-geometric
mujoco
opencv-python
tqdm
```

可先创建虚拟环境，再安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy mujoco opencv-python tqdm
```

PyTorch 与 PyTorch Geometric 建议参考各自官方命令安装，以匹配本机 CUDA 版本。

## 快速运行

运行视觉目标估计演示：

```bash
python run_vision.py
```

运行三个复合任务示例：

```bash
python subtask/task1_bt_demo.py
python subtask/task2_bt_demo.py
python subtask/task3_bt_demo.py
```

任务运行后，默认会在 `robot_command_output/` 下生成对应的 `movej(...)` 指令文件：

```text
task1_movej_commands.txt
task2_movej_commands.txt
task3_movej_commands.txt
```

## GNN 模型流程

数据生成、训练和求解相关代码位于 `GNN/`：

```bash
python GNN/generate_ur10e_data.py
python GNN/train.py
```

已训练模型权重与配套数据文件位于：

```text
GNN/best_gat_model.pt
GNN/ur10e_paired_data.npy
GNN/ik_stats.json
```

任务脚本通过 `GNN/ik_solver_api.py` 调用求解器，并结合 MuJoCo 正运动学验证收敛结果。

## 任务说明

| 脚本 | 功能 |
| --- | --- |
| `subtask/task1_bt_demo.py` | 视觉定位目标后执行抓取与转移 |
| `subtask/task2_bt_demo.py` | 围绕视觉目标执行巡检扫描 |
| `subtask/task3_bt_demo.py` | 执行抓取、装配与旋进动作 |

各任务的关键参数集中写在脚本开头，包括初始关节角、目标偏移、插补步数、速度参数和指令输出路径，便于实验时快速调整。

## 真实机器人迁移

当前仓库默认以文本方式记录机器人运动指令。`robot_ip_client.py` 保留了 TCP 发送适配接口，可在完成网络、安全边界和真实机器人联调后，将生成的 URScript 风格指令接入实际控制流程。

更多迁移说明见：

```text
real_robot_ip_migration_guide.md
docs/ur10e_real_robot_ip_control_guide.html
```

## 备注

- 运行 MuJoCo viewer 需要本机具备可用图形环境。
- 若修改模型路径、场景文件或关节名称，请同步检查 `GNN/ik_solver_api.py` 与任务脚本中的配置。
- 真实机器人实验前应先在仿真中验证完整轨迹，并确认速度、加速度、工作空间和碰撞风险。
