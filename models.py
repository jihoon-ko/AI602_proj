import torch
from torch_geometric.nn import GCNConv
from torch import nn
import torch.nn.functional as F
import torch_scatter
from itertools import chain

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, D, K, k_1):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList([GCNConv(hidden_dim if i > 0 else input_dim, hidden_dim if i < num_layers - 1 else output_dim) for i in range(num_layers)])
        self.quantization = Quantization(hidden_dim, D, K, k_1)
        
    def forward(self, data, tau=1.):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.quantization(x, tau)
        x = self.convs[-1](x, edge_index)
        return x
    
class Quantization(torch.nn.Module):
    def __init__(self, hidden_dim, D, K, k_1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.D = D
        self.K = K
        self.k_1 = k_1
        self.register_buffer('k_inds', torch.LongTensor(list(chain.from_iterable([[j for _ in range(k_1 << j)] for j in range(K)]))))
        self.centroids = nn.Parameter(torch.FloatTensor(k_1 * ((1 << K) - 1), hidden_dim))
        self.fc = nn.Linear(K, 1, bias=False)
        
        torch.nn.init.kaiming_normal_(self.centroids)
        
    def forward(self, H, tau):
        # shape of H : (#nodes, #hiddendim)
        num_nodes, num_ks = H.shape[0], self.centroids.shape[0]
        products = H.unsqueeze(1) * self.centroids.unsqueeze(0) # (#nodes, 1, #hiddendim) * (1, #ks, #hiddendim) -> (#nodes, #ks, #hiddendim)
        products = torch.sum(products.view(num_nodes, num_ks, self.D, -1), dim=-1) # (#nodes, #ks, #D)
        probs = torch_scatter.scatter_softmax(products / tau, self.k_inds, dim=1) # (#nodes, #ks, #D) -> (#nodes, #ks, #D)
        normalized_prods = probs.unsqueeze(-1) * self.centroids.view(1, num_ks, self.D, -1) # (#nodes, #ks, #D, #hiddendim/D)
        weighted_sums = torch_scatter.scatter(normalized_prods, self.k_inds, dim=1, dim_size=self.K, reduce='sum')
        quantized_feats = weighted_sums.view(num_nodes, self.K, -1)
        return self.fc(quantized_feats.permute(0, 2, 1)).squeeze(-1) / torch.sum(self.fc.weight.data)