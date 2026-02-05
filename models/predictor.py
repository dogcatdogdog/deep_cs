import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class EdgeMetricNet(nn.Module):
    """
    [DeepCS V3.2 Component]
    Predicts edge scaling factors (Beta).
    """
    def __init__(self, in_channels, hidden_channels):
        super(EdgeMetricNet, self).__init__()
        self.input_dim = 2 * in_channels
        
        self.lin1 = nn.Linear(self.input_dim, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        
        # Softplus ensures positive output. 
        self.act = nn.Softplus()
        
        # Init: Start with beta around 1.0
        # Softplus(0.55) approx 1.0
        nn.init.constant_(self.lin2.bias, 0.55)

    def forward(self, h, mask=None):
        # h: [B, N, d]
        # mask: [B, N]
        
        # Pairwise features: Sum and Diff
        h_i = h.unsqueeze(2)
        h_j = h.unsqueeze(1)
        # Concatenate symmetric features
        pair_feat = torch.cat([h_i + h_j, torch.abs(h_i - h_j)], dim=-1)
        
        x = F.relu(self.lin1(pair_feat))
        x = self.lin2(x)
        
        # Raw Beta (Always positive)
        beta = self.act(x).squeeze(-1) 
        
        if mask is not None:
            # Zero out invalid areas
            edge_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
            beta = beta * edge_mask.float()
            
        return beta

class GraphPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4, dropout=0.0):
        super(GraphPredictor, self).__init__()
        
        # Encoder
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Decoder
        self.edge_metric_net = EdgeMetricNet(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        # Node Embedding
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x 

    def predict_beta(self, h_dense, mask=None):
        # Interface for Training Loop
        return self.edge_metric_net(h_dense, mask)