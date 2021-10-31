import argparse
import torch
from torch_geometric.datasets import Planetoid

from models import *

def main(args):
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() and args.gpu > -1 else 'cpu')
    dataset = Planetoid(root=f'/tmp/{args.dataset}', name=args.dataset)
    data = dataset[0].to(device)
    
    model = GCN(dataset.num_node_features, args.hidden_dim, dataset.num_classes, args.num_layers).to(device)
    print(model)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    model.train()
    for epoch in range(args.num_epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print('Accuracy: {:.4f}'.format(acc))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset", type=str, default='cora', help="dataset name (cora/citeseer/pubmed)")
    parser.add_argument("--model", type=str, default='gcn', help="model name")
    parser.add_argument("--hidden-dim", type=int, default=16, help="hidden dimension for model")
    parser.add_argument("--num-layers", type=int, default=2, help="number of the layers")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="regularization (weight decay)")
    parser.add_argument("--num-epochs", type=int, default=200, help="the number of training epochs")
    parser.add_argument("--gpu", type=int, default=3, help="gpu id")
    args=parser.parse_args()
    main(args)