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

        # Apply dropout
        x = F.dropout(x, p=0.5, training=self.training)

        # Apply RNN
        x = self.RNN(x.unsqueeze(1))[0].squeeze(1)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Apply normalization
        x = F.layer_norm(x, x.size()[1:])

        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 6. Apply a final classifier
        x = self.linear(x)

        return x


class GCN3(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN3, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.conv3 = GCNConv(hidden_channels // 2, hidden_channels // 4)
        self.conv4 = GCNConv(hidden_channels // 4, hidden_channels // 2)
        self.conv5 = GCNConv(hidden_channels // 2, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)
        # TODO: try another final classifier?
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply graph convolutions at the node level (keep node structure)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Readout to graph-level representation
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.layer_norm(x, x.size()[1:]) # TODO: consider removing, maybe batch norm is better?

        # Final classifier
        x = self.linear(x)

        return x