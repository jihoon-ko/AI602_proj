import argparse
import torch
from torch_geometric.datasets import Planetoid

from models import *

def main(args):
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() and args.gpu > -1 else 'cpu')
    dataset = Planetoid(root=f'/tmp/{args.dataset}', name=args.dataset)
    data = dataset[0].to(device)
    
    model = GCN(dataset.num_node_features, args.hidden_dim, dataset.num_classes, args.num_layers, D=args.D, K=args.K, k_1=args.K1).to(device)
    print(model)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    model.train()
    max_acc = -1
    max_acc_val = -1
    for epoch in range(args.num_epochs):
        optimizer.zero_grad()
        out, reg1 = model(data, args.tau)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask]) + args.lmbda * reg1
        loss.backward()
        optimizer.step()
        
        model.eval()
        pred = model(data)[0].argmax(dim=1)
        correct_val = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        correct_test = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc_val = int(correct_val) / int(data.val_mask.sum())
        acc_test = int(correct_test) / int(data.test_mask.sum())
        if acc_val >= max_acc_val:
            torch.save(model.state_dict(), 'models/dim_{}_nl_{}_K_{}_K1_{}_D_{}_tau_{}_lmbda_{}.pkt'.format(args.hidden_dim, args.num_layers, args.K, args.K1, args.D, args.tau, args.lmbda))
            max_acc_val, max_acc = acc_val, acc_test
        print('Valid Accuracy: {:.3f}\t(Best Test Accuracy: {:.3f})'.format(acc_val, max_acc))
    
    with open('log.txt', 'a') as f:
        f.write('dim_{}_nl_{}_K_{}_K1_{}_D_{}_tau_{}_lmbda_{}.txt'.format(args.hidden_dim, args.num_layers, args.K, args.K1, args.D, args.tau, args.lmbda) + '\n')
        f.write(str(max_acc) + '\n\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # Global hyperparameters
    parser.add_argument("--dataset", type=str, default='cora', help="dataset name (cora/citeseer/pubmed)")
    parser.add_argument("--model", type=str, default='gcn', help="model name")
    parser.add_argument("--hidden-dim", type=int, default=16, help="hidden dimension for model")
    parser.add_argument("--num-layers", type=int, default=2, help="number of the layers")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="regularization (weight decay)")
    parser.add_argument("--num-epochs", type=int, default=300, help="the number of training epochs")
    parser.add_argument("--gpu", type=int, default=7, help="gpu id")
    
    # Hyperparameters for the proposed model
    parser.add_argument("--K", type=int, default=5, help="number of clusters")
    parser.add_argument("--K1", type=int, default=4, help="size of cluster 1")
    parser.add_argument("--D", type=int, default=4, help="number of division?")
    parser.add_argument("--tau", type=float, default=1.0, help="softmax temperature")
    parser.add_argument("--lmbda", type=float, default=1.0, help="regularization parameter")
    args=parser.parse_args()
    main(args)