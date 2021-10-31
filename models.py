import torch
from torch_geometric.nn import GCNConv
from torch import nn
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList([GCNConv(hidden_dim if i > 0 else input_dim, hidden_dim if i < num_layers - 1 else output_dim) for i in range(num_layers)])
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x