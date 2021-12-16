import torch
from torch_geometric.nn import GCNConv
from torch import nn
import torch.nn.functional as F
import torch_scatter
from itertools import chain

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, D, L, K, model_type=0):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList([GCNConv(hidden_dim if i > 0 else input_dim, hidden_dim if i < num_layers - 1 else output_dim) for i in range(num_layers)])
        self.quants = torch.nn.ModuleList([Quantization(hidden_dim, D, L, K) for i in range(num_layers)])
        self.model_type = model_type
        
    def forward(self, data, tau=1.0):
        x, edge_index = data.x, data.edge_index
        x_0, reg = self.encoder(x), 0
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.leaky_relu(x, 0.2)
            x = F.dropout(x, p=0.3, training=self.training)
            if self.model_type == 1:
                _x, _reg = self.quants[i](x, tau)
                x = x + _x
                reg += _reg
            elif self.model_type == 2:
                x = x + x_0
            elif self.model_type == 3:
                _x, _reg = self.quants[i](x, tau)
                x = x + _x + x_0
                reg += _reg

        x = self.convs[-1](x, edge_index)
        return x, reg
    
class Quantization(torch.nn.Module):
    def __init__(self, hidden_dim, D, L, K):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.D = D
        self.L = L
        self.K = K
        self.register_buffer('k_inds', torch.LongTensor(list(chain.from_iterable([[j for _ in range(K << j)] for j in range(L)]))))
        self.register_buffer('k_blns', torch.FloatTensor(list(chain.from_iterable([[1.0 / (K << j) for _ in range(K << j)] for j in range(L)]))))
        self.centroids = nn.Parameter(torch.FloatTensor(K * ((1 << L) - 1), hidden_dim))
        self.fc = nn.Linear(L, 1, bias=True)
        
        torch.nn.init.kaiming_normal_(self.centroids)
        
    def forward(self, H, tau=1.0):
        # shape of H : (#nodes, #hiddendim)
        num_nodes, num_ks = H.shape[0], self.centroids.shape[0]
        products = H.unsqueeze(1) * self.centroids.unsqueeze(0) # (#nodes, 1, #hiddendim) * (1, #ks, #hiddendim) -> (#nodes, #ks, #hiddendim)
        products = torch.sum(products.view(num_nodes, num_ks, self.D, -1), dim=-1) # (#nodes, #ks, #D)
        _, hard_indices = torch_scatter.scatter_max(products / tau, self.k_inds, dim=1) # (#nodes, #K, #D)
        
        hard_feats = torch.gather(self.centroids.repeat(num_nodes, 1, 1), dim=1, index=hard_indices.repeat_interleave(self.hidden_dim // self.D, dim=-1))
        soft_probs = torch_scatter.scatter_softmax(products / tau, self.k_inds, dim=1) # (#nodes, #ks, #D) -> (#nodes, #ks, #D)
        
        normalized_prods = soft_probs.unsqueeze(-1) * self.centroids.view(1, num_ks, self.D, -1) # (#nodes, #ks, #D, #hiddendim/D)
        weighted_sums = torch_scatter.scatter(normalized_prods, self.k_inds, dim=1, dim_size=self.L, reduce='sum')
        soft_feats = weighted_sums.view(num_nodes, self.L, -1)
        
        quantized_feats = (hard_feats - soft_feats).detach() + soft_feats
        
        balance = torch.sum(soft_probs, 0) / num_nodes
        target_balance = self.k_blns.repeat(self.D).view(self.D, -1).T
        reg = torch.norm(balance - target_balance, p='fro')
        
        pred = torch.max(quantized_feats, dim=1)[0]
        
        return pred, reg
