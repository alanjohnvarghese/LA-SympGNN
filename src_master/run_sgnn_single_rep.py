import argparse
from data import get_dataset
from heterophilic import get_fixed_splits
import numpy as np
import torch
from la_sgnn import SGNN

def train(model, optimizer, data):
    lf = torch.nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    out = model(data.x,data.edge_index)
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, data, opt):
    lf = torch.nn.CrossEntropyLoss()

    model.eval()
    out = model(data.x, data.edge_index)

    pred = out[data.train_mask]
    train_acc = (data.y.squeeze()[data.train_mask] == pred.argmax(dim=1)).sum().item()/opt["train_nodes"]
    
    pred = out[data.val_mask]
    val_acc   = (data.y.squeeze()[data.val_mask] == pred.argmax(dim=1)).sum().item()/opt["val_nodes"]
    val_loss = lf(out[data.val_mask], data.y.squeeze()[data.val_mask]).item()    

    pred = out[data.test_mask]
    test_acc  = (data.y.squeeze()[data.test_mask] == pred.argmax(dim=1)).sum().item()/opt["test_nodes"]

    return train_acc, val_acc, test_acc, val_loss 


def main(opt):

    torch.manual_seed(opt["seed"])
    
    dataset = get_dataset(opt,'../data',True)
    if opt['geom_gcn_splits']:
        data = get_fixed_splits(dataset.data, opt['dataset'],opt['rep'])

    dataset.data = data

    opt["train_nodes"] = torch.sum(dataset.data.train_mask).item()
    opt["val_nodes"]   = torch.sum(dataset.data.val_mask).item()
    opt["test_nodes"]  = torch.sum(dataset.data.test_mask).item()

    _,opt["num_features"] = dataset.data.x.shape
    opt["num_classes"] = max(dataset.data.y).item() + 1

    print(f'Finished processing {opt["dataset"]}')
    print(f'Number of training nodes: {opt["train_nodes"]}')
    print(f'Number of validation nodes: {opt["val_nodes"]}')
    print(f'Number of testing nodes: {opt["test_nodes"]}')
    print(f'Number of features:{opt["num_features"]}')
    print(f'Number of classes:{opt["num_classes"]}')
    print('\n')
    print(opt)

    model = SGNN(opt)
    optimizer = torch.optim.Adam(model.parameters(), lr = opt["lr"], weight_decay = opt["weight_decay"])
    
    train_acc = []
    val_acc = []
    test_acc = []

    train_l = []
    val_l = []

    for epoch in range(opt['epochs']):
        #start_time = time.time()
        loss = train(model, optimizer, data)
        train_l.append(loss)

        tmp_train_acc, tmp_val_acc, tmp_test_acc, tmp_val_loss = test(model, data, opt)

        train_acc.append(tmp_train_acc)
        val_acc.append(tmp_val_acc)
        test_acc.append(tmp_test_acc)
        val_l.append(tmp_val_loss)

        print(f'Epoch: {epoch}, train_loss:{loss:.4f}, '
              f'train_acc: {tmp_train_acc:.4f}, val_acc:{tmp_val_acc:.4f}, test_acc: {tmp_test_acc:.4f}')
        
    argmax = np.argmin(val_l)
    print(f"Max test acc is {test_acc[argmax]} at epoch {argmax}")
        
    


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()

    #Training settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

    parser.add_argument('--num_h_layers', type=int, default=0, help ='LAup,LAdown,ActUp, ActDown')
    parser.add_argument('--hidden', type=int, default=16, help='Hidden dimension.')
    parser.add_argument('--dataset', type=str, default='Cora', help='Cora,Citeseer,Pubmed,Film...' )
    
    parser.add_argument('--geom_gcn_splits', type=str, default='True', help='Use geom-gcn split')
    parser.add_argument('--rewiring', type=str, default=None, help='Graph rewiring.')
    parser.add_argument('--rep', type=int, default=0, help='Rep number in geom-gcn-split')

    args = parser.parse_args()

    opt = vars(args)
    main(opt)


