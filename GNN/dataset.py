import numpy as np
import torch
from torch_geometric.data import Dataset, Data


class IKDataset(Dataset):
    def __init__(self, data_file, normalize=True, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data = np.load(data_file)
        self.num_joints = 6
        self.normalize = normalize

        src = list(range(self.num_joints - 1))
        dst = list(range(1, self.num_joints))
        self.edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

        if self.normalize:
            self.mean, self.std = self._compute_node_stats()
            self.mean_np = self.mean.numpy()
            self.std_np = self.std.numpy()
        else:
            self.mean, self.std, self.mean_np, self.std_np = None, None, None, None

    def _compute_node_stats(self):
        print("正在预计算特征标准化系数...")
        all_node_feats = []
        sample_size = min(len(self.data), 5000)
        for i in range(sample_size):
            row = self.data[i]
            cur_angles = row[:6]
            cur_pos_all = row[6:24].reshape(6, 3)
            cur_rot_all = row[24:78].reshape(6, 9)
            target_pos = row[78:81]
            target_rot = row[81:90]

            for j in range(self.num_joints):
                rel_pos = target_pos - cur_pos_all[j]
                rel_rot = target_rot - cur_rot_all[j]
                # 🔥 修改：将单标量角度改为 sin 和 cos
                feat = np.concatenate(
                    [[np.sin(cur_angles[j]), np.cos(cur_angles[j])], cur_pos_all[j], cur_rot_all[j], rel_pos, rel_rot])
                all_node_feats.append(feat)

        all_node_feats = np.array(all_node_feats)
        mean = np.mean(all_node_feats, axis=0)
        std = np.std(all_node_feats, axis=0)
        std[std < 1e-6] = 1.0
        return torch.from_numpy(mean).float(), torch.from_numpy(std).float()

    def len(self):
        return len(self.data)

    def get(self, idx):
        row = self.data[idx]

        cur_angles = row[:6]
        cur_pos_all = row[6:24].reshape(6, 3)
        cur_rot_all = row[24:78].reshape(6, 9)
        target_pos = row[78:81]
        target_rot = row[81:90]
        delta_y = row[90:96]

        node_list = []
        for i in range(self.num_joints):
            idx_onehot = np.zeros(6, dtype=np.float32)
            idx_onehot[i] = 1.0

            rel_pos = target_pos - cur_pos_all[i]
            rel_rot = target_rot - cur_rot_all[i]

            # 🔥 修改：将单标量角度改为 sin 和 cos
            num_feat = np.concatenate(
                [[np.sin(cur_angles[i]), np.cos(cur_angles[i])], cur_pos_all[i], cur_rot_all[i], rel_pos, rel_rot])
            if self.normalize:
                num_feat = (num_feat - self.mean_np) / self.std_np

            final_feat = np.concatenate([idx_onehot, num_feat])
            node_list.append(final_feat)

        x = torch.from_numpy(np.array(node_list)).float()
        y = torch.from_numpy(delta_y).float()

        # 🔥 核心修正：将 cur_angles 原封不动带入网络！
        cur_angles_t = torch.from_numpy(cur_angles).float()

        return Data(x=x, edge_index=self.edge_index, y=y, cur_angles=cur_angles_t)