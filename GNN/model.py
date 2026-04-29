import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class IKGATModel(nn.Module):
    # 🔥 修改：in_channels 默认值从 31 变为 32
    def __init__(self, in_channels=32, hidden_channels=256, out_channels=1,
                 num_layers=4, heads=4, dropout=0.0):
        super(IKGATModel, self).__init__()
        self.num_joints = 6
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))

        self.skip = nn.Linear(in_channels, hidden_channels * heads) if in_channels != hidden_channels * heads else nn.Identity()

        # 🔥 核心升级：强大的非线性 MLP 解码头 🔥
        # GNN 提取完图特征后，必须通过 MLP 进行反三角函数的非线性逼近
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_channels * heads, hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x_input = x
        x = self.convs[0](x, edge_index)
        x = self.norms[0](x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = x + self.skip(x_input)

        for i in range(1, len(self.convs)):
            x_input = x
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = x + x_input

        # 使用 MLP 替代原先单薄的 GATv2Conv
        x = self.mlp_head(x)

        if hasattr(data, 'num_graphs') and data.num_graphs is not None:
            batch_size = data.num_graphs
            out = x.view(batch_size, self.num_joints)
        elif hasattr(data, 'batch') and data.batch is not None:
            batch_size = int(data.batch.max()) + 1
            out = x.view(batch_size, self.num_joints)
        else:
            out = x.view(-1, self.num_joints)
        return out