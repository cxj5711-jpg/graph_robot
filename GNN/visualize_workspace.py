import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D


def visualize_workspace(data_file, max_points=10000):
    print(f"正在加载数据文件: {data_file}...")
    try:
        data = np.load(data_file)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {data_file}，请检查路径。")
        return

    print(f"✅ 数据集加载成功！总样本数: {data.shape[0]}, 特征维度: {data.shape[1]}")

    # 解析数据格式
    # cur_angles(6) + cur_poses(18) + cur_mats(54) + tgt_ee_pos(3) + tgt_ee_rot(9) + delta(6) = 96
    # 目标末端位置 (Target End-Effector Position) 在索引 78, 79, 80
    tgt_pos = data[:, 78:81]

    # 为了防止画图卡顿，如果点数过多则进行随机下采样
    if len(tgt_pos) > max_points:
        print(f"为了保证渲染流畅度，将随机抽取 {max_points} 个点进行展示...")
        indices = np.random.choice(len(tgt_pos), max_points, replace=False)
        plot_pos = tgt_pos[indices]
    else:
        plot_pos = tgt_pos

    x = plot_pos[:, 0]
    y = plot_pos[:, 1]
    z = plot_pos[:, 2]

    # 绘制 3D 散点图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 按照 Z 轴高度 (高度) 进行着色，方便观察空间层次
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=2, alpha=0.6)

    # 绘制 x=0.2 和 z=0.2 的约束参考面/线
    ax.plot([0.2, 0.2], [min(y), max(y)], [min(z), min(z)], color='red', linestyle='--', linewidth=2,
            label='x=0.2 Constraint')

    ax.set_xlabel('X Axis (m)', fontweight='bold')
    ax.set_ylabel('Y Axis (m)', fontweight='bold')
    ax.set_zlabel('Z Axis (m)', fontweight='bold')
    ax.set_title(f'UR10e Dataset Workspace Distribution\n(Visualizing {len(plot_pos)} Target Points)', fontsize=14)

    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Z Height (m)')

    # 限制坐标轴范围以便观察 x>0.2, z>0.2 的分布切面
    ax.set_xlim([min(x) - 0.1, max(x) + 0.1])
    ax.set_ylim([min(y) - 0.1, max(y) + 0.1])
    ax.set_zlim([min(z) - 0.1, max(z) + 0.1])

    plt.legend()
    plt.tight_layout()
    print("正在生成可视化窗口，请稍候...")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the 3D workspace of the generated UR10e dataset.")
    parser.add_argument("--data", type=str, default="ur10e_paired_data.npy", help="Path to the dataset .npy file")
    parser.add_argument("--samples", type=int, default=15000, help="Maximum number of points to plot")
    args = parser.parse_args()

    visualize_workspace(args.data, args.samples)