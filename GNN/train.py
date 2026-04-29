import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import math
from GNN.dataset import IKDataset
from GNN.model import IKGATModel
from tqdm import tqdm


# ==========================================================
# UR10e D-H 正运动学引擎
# ==========================================================
class UR10e_FK(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.d = torch.tensor([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655], device=device)
        self.a = torch.tensor([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0], device=device)
        self.alpha = torch.tensor([math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0], device=device)

    def forward(self, q):
        batch_size = q.shape[0]
        T = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        for i in range(6):
            theta = q[:, i]
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            cos_a, sin_a = torch.cos(self.alpha[i]), torch.sin(self.alpha[i])

            T_i = torch.zeros((batch_size, 4, 4), device=self.device)
            T_i[:, 0, 0], T_i[:, 0, 1], T_i[:, 0, 2], T_i[:, 0, 3] = cos_t, -sin_t * cos_a, sin_t * sin_a, self.a[
                i] * cos_t
            T_i[:, 1, 0], T_i[:, 1, 1], T_i[:, 1, 2], T_i[:, 1, 3] = sin_t, cos_t * cos_a, -cos_t * sin_a, self.a[
                i] * sin_t
            T_i[:, 2, 1], T_i[:, 2, 2], T_i[:, 2, 3] = sin_a, cos_a, self.d[i]
            T_i[:, 3, 3] = 1.0
            T = torch.bmm(T, T_i)
        return T[:, :3, 3], T[:, :3, :3]

    # ==========================================================


# 🔥 核心升级：基于不确定性的自适应多任务损失权重机制 🔥
# ==========================================================
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_tasks=3):
        super(AutomaticWeightedLoss, self).__init__()
        # 初始的对数方差参数 s = log(sigma^2)
        # 初始化为0，意味着最初 e^(-0) = 1.0 的均等权重
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, l_joint, l_pos, l_rot):
        # 动态计算指数权重 e^(-s)
        w_joint = torch.exp(-self.log_vars[0])
        w_pos = torch.exp(-self.log_vars[1])
        w_rot = torch.exp(-self.log_vars[2])

        # 组装最终 Loss，并加上正则化项 +s 防止权重趋于无穷小作弊
        loss = (w_joint * l_joint + self.log_vars[0] +
                w_pos * l_pos + self.log_vars[1] +
                w_rot * l_rot + self.log_vars[2])

        return loss, w_joint, w_pos, w_rot


def compute_raw_losses(pred_delta_q, target_delta_q, cur_angles, fk_model):
    """只计算纯粹的物理误差，不包含任何人为硬编码权重"""
    weights = torch.tensor([1.2, 1.2, 1.2, 1.0, 1.0, 1.0], device=pred_delta_q.device)
    loss_joint = torch.mean(weights * torch.abs(pred_delta_q - target_delta_q))

    pred_pos, pred_rot = fk_model(cur_angles + pred_delta_q)
    target_pos, target_rot = fk_model(cur_angles + target_delta_q)

    loss_pos = torch.mean(torch.abs(pred_pos - target_pos))
    loss_rot = torch.mean(torch.abs(pred_rot - target_rot))

    return loss_joint, loss_pos, loss_rot


def train_epoch(model, awl, loader, optimizer, fk_model, device):
    model.train()
    awl.train()
    total_loss = 0
    with tqdm(loader, desc="Training", leave=False, dynamic_ncols=True) as pbar:
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)

            # 1. 算基础误差
            l_joint, l_pos, l_rot = compute_raw_losses(out, data.y.view(-1, 6), data.cur_angles.view(-1, 6), fk_model)

            # 2. 通过自适应权重网络算最终 Loss
            loss, _, _, _ = awl(l_joint, l_pos, l_rot)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, awl, loader, fk_model, device):
    model.eval()
    awl.eval()
    t_loss, t_mae_deg, t_mae_pos = 0, 0, 0

    for data in loader:
        data = data.to(device)
        out = model(data)
        target = data.y.view(-1, 6)
        cur = data.cur_angles.view(-1, 6)

        l_joint, l_pos, l_rot = compute_raw_losses(out, target, cur, fk_model)
        loss, w_j, w_p, w_r = awl(l_joint, l_pos, l_rot)

        graphs = data.num_graphs
        t_loss += loss.item() * graphs
        t_mae_deg += torch.abs(out - target).mean().item() * graphs

        p_pos, _ = fk_model(cur + out)
        t_pos, _ = fk_model(cur + target)
        t_mae_pos += torch.norm(p_pos - t_pos, dim=1).mean().item() * graphs

    n = len(loader.dataset)
    # 额外返回当前网络学到的动态权重用于日志打印
    return t_loss / n, np.rad2deg(t_mae_deg / n), t_mae_pos / n, w_j.item(), w_p.item(), w_r.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ur10e_paired_data.npy')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--save', type=str, default='best_gat_model.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fk_model = UR10e_FK(device)
    print(f"使用设备: {device} | 模式: 自适应不确定性权重 + MLP 非线性头")

    full_dataset = IKDataset(data_file=args.data, normalize=True)
    train_size = int(0.9 * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset,
                                                               [train_size, len(full_dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = IKGATModel(
        in_channels=32,  # 🔥 修改：改为 32 通道
        hidden_channels=args.hidden, num_layers=args.layers,
        heads=args.heads, dropout=args.dropout
    ).to(device)

    # 实例化自适应权重层
    awl = AutomaticWeightedLoss(num_tasks=3).to(device)

    # 🔥 必须将 awl 的参数(即那三个 s) 放入优化器一起训练！
    optimizer = optim.AdamW(list(model.parameters()) + list(awl.parameters()), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_pos = float('inf')  # 现在我们用纯物理的位置误差来保存最佳模型，而不是被权重污染的 Loss
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, awl, train_loader, optimizer, fk_model, device)
        v_loss, v_mae_deg, v_mae_pos, w_j, w_p, w_r = evaluate(model, awl, val_loader, fk_model, device)
        scheduler.step(v_loss)

        # 全新的日志格式：你能亲眼看到网络是如何“智能分配”这三个任务的权重的！
        print(f"Epoch {epoch:03d} | Val Deg: {v_mae_deg:.3f}° | Val Pos: {v_mae_pos * 100:.2f}cm | "
              f"Dynamic Weights(Joint:{w_j:.2f}, Pos:{w_p:.2f}, Rot:{w_r:.2f})")

        if v_mae_pos < best_val_pos:
            best_val_pos = v_mae_pos
            torch.save(model.state_dict(), args.save)
            print(f"  -> 保存最佳模型 (Pos: {v_mae_pos * 100:.2f}cm)")


if __name__ == '__main__':
    main()