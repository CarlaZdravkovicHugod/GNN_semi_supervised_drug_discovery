import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, DenseGCNConv, DenseGraphConv, AGNNConv, GINConv, SAGEConv, GATConv, global_add_pool, GINEConv
import torch_geometric
import torch.nn as nn


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
    

class GCN4(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN4, self).__init__()
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

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = self.conv4(x, edge_index)
        x = F.relu(x)

        x = self.conv5(x, edge_index)
        x = F.relu(x)

        # Readout to graph-level representation
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.layer_norm(x, x.size()[1:]) # TODO: consider removing, maybe batch norm is better?

        # Final classifier
        x = self.linear(x)

        return x
    
class GCN5(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN5, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply graph convolutions at the node level (keep node structure)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = self.conv4(x, edge_index)
        x = F.relu(x)

        x = self.conv5(x, edge_index)
        x = F.relu(x)

        # Readout to graph-level representation
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.layer_norm(x, x.size()[1:]) # TODO: consider removing, maybe batch norm is better?

        # Final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)

        return x
    

class GCN6(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN6, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply graph convolutions at the node level (keep node structure)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Readout to graph-level representation
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # Final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)

        return x

class GCN7(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN7, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply graph convolutions at the node level (keep node structure)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[1:]) 

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[1:]) 

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[1:]) 

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[1:]) 
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Readout to graph-level representation
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.layer_norm(x, x.size()[1:]) # TODO: consider removing, maybe batch norm is better?

        # Final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)

        return x
    
class GCN8(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN8, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply graph convolutions at the node level (keep node structure)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
    
        for _ in range(5):
            x = self.conv2(x, edge_index)
            x = F.relu(x)
   
        # Readout to graph-level representation
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.layer_norm(x, x.size()[1:]) # TODO: consider removing, maybe batch norm is better?

        x = F.dropout(x, p=0.5, training=self.training)
        
        # Final classifier
        x = self.linear(x)

        return x
    
class GCN9(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN9, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply graph convolutions at the node level (keep node structure)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        x = F.layer_norm(x, x.size()[1:]) # TODO: consider removing, maybe batch norm is better?
        x = F.dropout(x, p=0.5, training=self.training)
    
        for _ in range(5):
            x = self.conv2(x, edge_index)
            x = F.relu(x)
   
        # Readout to graph-level representation
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.layer_norm(x, x.size()[1:]) # TODO: consider removing, maybe batch norm is better?

        x = F.dropout(x, p=0.5, training=self.training)
        
        # Final classifier
        x = self.linear(x)

        return x
    
class GCN10(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN10, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply graph convolutions at the node level (keep node structure)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        for _ in range(3):
            x = self.conv2(x, edge_index)
            x = F.relu(x)
   
        # Readout to graph-level representation
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.batch_norm(x, x.size()[1:])

        x = F.dropout(x, p=0.5, training=self.training)
        
        # Final classifier
        x = self.linear(x)

        return x
    
class GCN11(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN11, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.AGNconv1 = AGNNConv(hidden_channels)
        self.AGNconv2 = AGNNConv(hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply graph convolutions at the node level (keep node structure)
        x = self.conv1(x, edge_index)
        x = F.tanh(x)

        x = self.conv2(x, edge_index)
        x = F.tanh(x)

        x = self.conv3(x, edge_index)
        x = F.tanh(x)

        x = self.conv4(x, edge_index)
        x = F.tanh(x)

        x = self.conv5(x, edge_index)
        x = F.tanh(x)

        x = self.AGNconv1(x, edge_index)
        x = F.tanh(x)
        x = self.AGNconv2(x, edge_index)
        x = F.tanh(x)

        # Readout to graph-level representation
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.layer_norm(x, x.size()[1:])

        # Final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)

        return x

class GCN12(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN12, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.AGNconv1 = AGNNConv(hidden_channels)
        self.AGNconv2 = AGNNConv(hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply graph convolutions at the node level (keep node structure)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = self.conv4(x, edge_index)
        x = F.relu(x)

        x = self.conv5(x, edge_index)
        x = F.relu(x)

        x = self.AGNconv1(x, edge_index)
        x = F.relu(x)
        x = self.AGNconv2(x, edge_index)
        x = F.relu(x)

        # Readout to graph-level representation
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.layer_norm(x, x.size()[1:])

        # Final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)

        return x
    
class GCN13(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=256):
        super(GCN13, self).__init__()
        
        # Initial projection
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Graph conv layers with residual connections
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Multi-level readout (3x hidden_channels after concat)
        # MLP head
        self.fc1 = torch.nn.Linear(hidden_channels * 3, hidden_channels * 2)
        self.bn_fc1 = torch.nn.BatchNorm1d(hidden_channels * 2)
        
        self.fc2 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.bn_fc2 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.fc3 = torch.nn.Linear(hidden_channels, 1)
        
        self.dropout = 0.3
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2 with residual
        identity = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity  # Residual connection
        
        # Layer 3 with residual
        identity = x
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        
        # Layer 4 with residual
        identity = x
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        
        # Multi-level readout: combine mean, max, and sum pooling
        x_mean = global_mean_pool(x, batch)
        x_max = torch_geometric.nn.global_max_pool(x, batch)
        x_sum = torch_geometric.nn.global_add_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # MLP head
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc3(x)
        
        return x
    

from torch_geometric.nn import GATConv

class GAT14(torch.nn.Module):
    """
    Graph Attention Network with multi-head attention,
    residual connections, and robust readout
    """
    def __init__(self, num_node_features, hidden_channels=256, heads=4):
        super(GAT14, self).__init__()
        
        # GAT layers with multi-head attention
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=0.3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)
        
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.3)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * heads)
        
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.3)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        
        # MLP head with multi-level pooling
        self.fc1 = torch.nn.Linear(hidden_channels * 3, hidden_channels * 2)
        self.bn_fc1 = torch.nn.BatchNorm1d(hidden_channels * 2)
        
        self.fc2 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.bn_fc2 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.fc3 = torch.nn.Linear(hidden_channels, 1)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GAT layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        
        # Multi-level readout
        x_mean = global_mean_pool(x, batch)
        x_max = torch_geometric.nn.global_max_pool(x, batch)
        x_sum = torch_geometric.nn.global_add_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # MLP head
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.fc3(x)
        
        return x


class GIN15(torch.nn.Module):
    """
    Graph Isomorphism Network (GIN) with residual connections,
    batch normalization, and multi-level readout.
    GIN is more expressive than GCN for graph classification.
    """
    def __init__(self, num_node_features, hidden_channels=256):
        super(GIN15, self).__init__()
        
        # GIN layers with MLPs
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(num_node_features, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
        )
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
        )
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv3 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
        )
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv4 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
        )
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Multi-level readout (3x hidden_channels after concat)
        # MLP head
        self.fc1 = torch.nn.Linear(hidden_channels * 3, hidden_channels * 2)
        self.bn_fc1 = torch.nn.BatchNorm1d(hidden_channels * 2)
        
        self.fc2 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.bn_fc2 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.fc3 = torch.nn.Linear(hidden_channels, 1)
        
        self.dropout = 0.3
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2 with residual
        identity = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        
        # Layer 3 with residual
        identity = x
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        
        # Layer 4 with residual
        identity = x
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        
        # Multi-level readout: combine mean, max, and sum pooling
        x_mean = global_mean_pool(x, batch)
        x_max = torch_geometric.nn.global_max_pool(x, batch)
        x_sum = torch_geometric.nn.global_add_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # MLP head
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc3(x)
        
        return x


class GraphSAGE16(torch.nn.Module):
    """
    GraphSAGE model with mean aggregation, residual connections,
    batch normalization, and multi-level readout.
    GraphSAGE is efficient and works well for inductive learning.
    """
    def __init__(self, num_node_features, hidden_channels=256):
        super(GraphSAGE16, self).__init__()
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Multi-level readout (3x hidden_channels after concat)
        # MLP head
        self.fc1 = torch.nn.Linear(hidden_channels * 3, hidden_channels * 2)
        self.bn_fc1 = torch.nn.BatchNorm1d(hidden_channels * 2)
        
        self.fc2 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.bn_fc2 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.fc3 = torch.nn.Linear(hidden_channels, 1)
        
        self.dropout = 0.3
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2 with residual
        identity = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        
        # Layer 3 with residual
        identity = x
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        
        # Layer 4 with residual
        identity = x
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        
        # Multi-level readout: combine mean, max, and sum pooling
        x_mean = global_mean_pool(x, batch)
        x_max = torch_geometric.nn.global_max_pool(x, batch)
        x_sum = torch_geometric.nn.global_add_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # MLP head
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc3(x)
        
        return x
    


class GCN17(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128):
        super(GCN17, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply graph convolutions at the node level (keep node structure)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Readout to graph-level representation
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.layer_norm(x, x.size()[1:])

        # Final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)

        return x


class GCN18(torch.nn.Module):
    """
    GCN with broader layers (128 channels), batch normalization, 
    dropout, 3 GCN layers, and multi-level pooling (mean, max, sum).
    """
    def __init__(self, num_node_features, hidden_channels=128):
        super(GCN18, self).__init__()
        
        # 3 GCN layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Linear layer for final prediction (3x hidden_channels due to concat pooling)
        self.linear = torch.nn.Linear(hidden_channels * 3, 1)
        
        self.dropout = 0.5
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Multi-level pooling: combine mean, max, and sum pooling
        x_mean = global_mean_pool(x, batch)
        x_max = torch_geometric.nn.global_max_pool(x, batch)
        x_sum = torch_geometric.nn.global_add_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # Final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        
        return x
   
class GCN19(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128):
        super(GCN19, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)
        self.dropout = 0.5
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
       
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        
        return x
   # Try 18 without dropout at every layer
    
class GINE5(torch.nn.Module):
    def __init__(self, num_node_features, edge_dim=4, hidden_channels=64):
        super().__init__()

        # Shared MLP used by GINEConv
        def mlp():
            return torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )

        # First layer: project node features to hidden size
        self.lin_in = torch.nn.Linear(num_node_features, hidden_channels)

        # GINEConv layers (edge aware)
        self.conv1 = GINEConv(mlp(), edge_dim=edge_dim)
        self.conv2 = GINEConv(mlp(), edge_dim=edge_dim)
        self.conv3 = GINEConv(mlp(), edge_dim=edge_dim)
        self.conv4 = GINEConv(mlp(), edge_dim=edge_dim)
        self.conv5 = GINEConv(mlp(), edge_dim=edge_dim)

        # Final readout layer
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Project input features
        x = self.lin_in(x)
        x = F.relu(x)

        # Edge-aware conv layers
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = F.relu(self.conv4(x, edge_index, edge_attr))
        x = F.relu(self.conv5(x, edge_index, edge_attr))

        # Pool graph representation
        x = global_mean_pool(x, batch)

        # Layer norm
        x = F.layer_norm(x, x.size()[1:])

        # Dropout
        x = F.dropout(x, p=0.3, training=self.training)

        # Output
        return self.linear(x)