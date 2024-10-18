import torch_geometric.nn as pyg_nn
import torch.nn as nn
import torch 


class GNNModel(nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = pyg_nn.GraphConv(3, 128)
        self.conv2 = pyg_nn.GraphConv(128, 128)
        self.pool = pyg_nn.global_mean_pool
        self.fc1 = nn.Linear(128, 138)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch)
        x = self.fc1(x)
        return torch.sigmoid(x)
