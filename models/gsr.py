import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops


class GSR(nn.Module):
    """
    Graph Structure-Recognition (GSR) model for molecular property prediction
    using Graph Attention Networks (GAT).
    
    The model processes molecular graphs to predict properties like lipophilicity.
    It returns both node-level and graph-level predictions.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2):
        """
        Initialize the GSR model.
        
        Parameters:
        - in_channels: Size of input features for each atom
        - hidden_channels: Size of hidden layer features
        - out_channels: Size of output features (usually 1 for regression)
        - heads: Number of attention heads in GATConv
        - dropout: Dropout probability
        """
        super(GSR, self).__init__()
        
        # Multiple attention heads for better feature extraction
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.conv4 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        
        # Final output layer with single head
        self.conv_out = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, data):
        """
        Forward pass through the network.
        
        Parameters:
        - data: PyTorch Geometric Data object containing x (node features) and edge_index
        
        Returns:
        - node_preds: Predictions for each atom
        - graph_preds: Predictions for the entire molecule (via mean pooling)
        """
        x, edge_index = data.x, data.edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Apply the graph convolutional layers with batch normalization
        x = self.conv1(x, edge_index)
        x = torch.tanh(x)

        x = self.conv2(x, edge_index)
        x = torch.tanh(x)

        x = self.conv3(x, edge_index)
        x = torch.tanh(x)

        x = self.conv4(x, edge_index)
        x = torch.tanh(x)
        # Final output layer
        node_preds = self.conv_out(x, edge_index)
        
        # Global mean pooling to get a single prediction per graph
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(node_preds.size(0), dtype=torch.long, device=node_preds.device)
        graph_preds = global_mean_pool(node_preds, batch)
        
        return node_preds, graph_preds