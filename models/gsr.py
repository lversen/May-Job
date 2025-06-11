import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops


class GSR(nn.Module):
    """
    Graph Structure-Recognition (GSR) model for molecular property prediction
    using Graph Attention Networks (GAT) - TChemGNN from the paper.
    
    The model processes molecular graphs to predict properties like lipophilicity.
    It returns both node-level and graph-level predictions.
    
    Architecture from paper:
    - 5 GAT layers with hyperbolic tangent activation
    - Hidden channels: 28 (best results)
    - ~3.7K learnable parameters
    """
    
    def __init__(self, in_channels=36, hidden_channels=28, out_channels=1, heads=1, dropout=0):
        """
        Initialize the GSR model according to the paper specifications.
        
        Parameters:
        - in_channels: Size of input features for each atom (36 in paper)
        - hidden_channels: Size of hidden layer features (28 in paper)
        - out_channels: Size of output features (1 for regression)
        - heads: Number of attention heads in GATConv (1 in paper)
        - dropout: Dropout probability (0 in paper)
        """
        super(GSR, self).__init__()
        
        # 5 GAT layers as specified in the paper
        # Paper mentions "5 layers of Graph Attention Network"
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.conv4 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.conv5 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        
        # Output layer to produce single value per node
        # Paper doesn't use concat=False in intermediate layers but does for output
        self.conv_out = nn.Linear(hidden_channels * heads, out_channels)
        
        # Print model size for verification (~3.7K parameters as per paper)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total model parameters: {total_params}")

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

        # Apply the 5 GAT layers with hyperbolic tangent activation (as per paper)
        x = self.conv1(x, edge_index)
        x = torch.tanh(x)  # Paper specifies hyperbolic tangent

        x = self.conv2(x, edge_index)
        x = torch.tanh(x)

        x = self.conv3(x, edge_index)
        x = torch.tanh(x)

        x = self.conv4(x, edge_index)
        x = torch.tanh(x)
        
        x = self.conv5(x, edge_index)
        x = torch.tanh(x)
        
        # Final output layer (linear transformation to get predictions)
        node_preds = self.conv_out(x)
        
        # Global mean pooling to get a single prediction per graph
        # Note: The paper mentions that for ESOL and FreeSolv, they use the last node's prediction
        # instead of pooling, but we still provide both options here
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(node_preds.size(0), dtype=torch.long, device=node_preds.device)
        graph_preds = global_mean_pool(node_preds, batch)
        
        return node_preds, graph_preds


class GSRNoPooling(nn.Module):
    """
    GSR model variant that uses specific node predictions instead of pooling.
    According to the paper, this works better for ESOL and FreeSolv datasets.
    """
    
    def __init__(self, in_channels=36, hidden_channels=28, out_channels=1, heads=1, dropout=0):
        """
        Initialize the GSR no-pooling model.
        
        Parameters:
        - in_channels: Size of input features for each atom (36 in paper)
        - hidden_channels: Size of hidden layer features (28 in paper)  
        - out_channels: Size of output features (1 for regression)
        - heads: Number of attention heads in GATConv (1 in paper)
        - dropout: Dropout probability (0 in paper)
        """
        super(GSRNoPooling, self).__init__()
        
        # Same architecture as GSR
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.conv4 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.conv5 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        
        # Output layer
        self.conv_out = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data, node_selection='last'):
        """
        Forward pass through the network.
        
        Parameters:
        - data: PyTorch Geometric Data object containing x (node features) and edge_index
        - node_selection: Which node(s) to use for prediction ('last', 'first', or index)
        
        Returns:
        - node_preds: Predictions for each atom
        - graph_preds: Predictions based on selected node(s)
        """
        x, edge_index = data.x, data.edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Apply the 5 GAT layers with hyperbolic tangent activation
        x = self.conv1(x, edge_index)
        x = torch.tanh(x)

        x = self.conv2(x, edge_index)
        x = torch.tanh(x)

        x = self.conv3(x, edge_index)
        x = torch.tanh(x)

        x = self.conv4(x, edge_index)
        x = torch.tanh(x)
        
        x = self.conv5(x, edge_index)
        x = torch.tanh(x)
        
        # Final output layer
        node_preds = self.conv_out(x)
        
        # Select specific nodes for graph-level prediction
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(node_preds.size(0), dtype=torch.long, device=node_preds.device)
        
        # Get the number of nodes per graph
        batch_size = batch.max().item() + 1
        graph_preds = []
        
        for i in range(batch_size):
            # Get nodes for this graph
            mask = batch == i
            graph_node_preds = node_preds[mask]
            
            if node_selection == 'last':
                # Use the last node (as per paper for ESOL/FreeSolv)
                selected_pred = graph_node_preds[-1]
            elif node_selection == 'first':
                # Use the first node
                selected_pred = graph_node_preds[0]
            elif isinstance(node_selection, int):
                # Use specific index
                selected_pred = graph_node_preds[node_selection]
            else:
                # Default to mean pooling
                selected_pred = graph_node_preds.mean(dim=0)
            
            graph_preds.append(selected_pred)
        
        graph_preds = torch.stack(graph_preds)
        
        return node_preds, graph_preds