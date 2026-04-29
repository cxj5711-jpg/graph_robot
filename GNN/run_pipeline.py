import subprocess
import sys


def run_command(cmd, description):
    print(f"\n=== {description} ===")
    print(f"执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"步骤失败，退出码 {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"{description} 完成")


if __name__ == "__main__":
    # 扩大数据规模到 40000，补充多尺度采样所需的数据量
    num_samples = 40000
    data_file = "ur10e_paired_data.npy"

    # 增加 Epochs
    epochs = 200
    model_save = "best_gat_model.pt"

    hidden = 256
    layers = 4
    heads = 4

    # 数值回归任务里先保留较低 dropout
    dropout = 0.05

    run_command(
        f"python generate_ur10e_data.py --num_samples {num_samples} --output {data_file}",
        "生成多尺度配对数据集"
    )

    run_command(
        f"python train.py --data {data_file} --epochs {epochs} --hidden {hidden} --layers {layers} --heads {heads} --dropout {dropout} --save {model_save}",
        "训练 GAT 模型"
    )

    run_command(
        f"python evaluate.py --model_path {model_save} --data_path {data_file} --hidden {hidden} --layers {layers} --heads {heads}",
        "可视化评估模型"
    )

    print("\n整条训练与评估流水线运行完成。")
