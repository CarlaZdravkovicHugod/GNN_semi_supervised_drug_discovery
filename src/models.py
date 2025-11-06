import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, DenseGCNConv, DenseGraphConv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.linear(x)

        return x


class GCN2(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.RNN = torch.nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        self.linear = torch.nn.Linear(hidden_channels, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # Apply dropout
        x = F.dropout(x, p=0.5, training=self.training)

        # Apply RNN
        x = self.RNN(x.unsqueeze(1))[0].squeeze(1)
        x = F.relu(x)

        # Apply normalization
        x = F.layer_norm(x, x.size()[1:])

        # 6. Apply a final classifier
        x = self.linear(x)

        return x
