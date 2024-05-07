import argparse
from data import get_dataset
from heterophilic import get_fixed_splits
import numpy as np
import torch
from la_sgnn import SGNN, SGNN_NoInteraction, SGNN_version2, SGNN_version3, SGNN_Attention, SGNN_source
import os
import wandb

def train_func(model, optimizer, data):
    lf = torch.nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    out = model(data.x,data.edge_index)
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


def test_func(model, data, opt):
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


def train_sgnn():

    run  = wandb.init()
    

    opt = {
        "seed": wandb.config.seed,
        "epochs": wandb.config.epochs,
        "lr": wandb.config.lr,
        "weight_decay": wandb.config.weight_decay,
        "num_h_layers": wandb.config.num_h_layers,
        "hidden": wandb.config.hidden,
        "input_dropout": wandb.config.input_dropout,
        "dataset": wandb.config.dataset,  # Example fixed parameter
        "geom_gcn_splits": wandb.config.geom_gcn_splits,
        "rewiring": wandb.config.rewiring # Example fixed parameter
    }

    torch.manual_seed(opt["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')

    best_val_acc = []
    best_test_acc = []

    for rep in range(10):
        dataset = get_dataset(opt,'../data',True)
        if opt['geom_gcn_splits']:
            data = get_fixed_splits(dataset.data, opt['dataset'],rep)

        dataset.data = data

        opt["train_nodes"] = torch.sum(dataset.data.train_mask).item()
        opt["val_nodes"]   = torch.sum(dataset.data.val_mask).item()
        opt["test_nodes"]  = torch.sum(dataset.data.test_mask).item()

        _,opt["num_features"] = dataset.data.x.shape
        opt["num_classes"] = max(dataset.data.y).item() + 1

        # print(f'Finished processing {opt["dataset"]}')
        # print(f'Number of training nodes: {opt["train_nodes"]}')
        # print(f'Number of validation nodes: {opt["val_nodes"]}')
        # print(f'Number of testing nodes: {opt["test_nodes"]}')
        # print(f'Number of features:{opt["num_features"]}')
        # print(f'Number of classes:{opt["num_classes"]}')
        # print('\n')
        # print(opt)

        data = data.to(device)
        model = SGNN_source(opt).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = opt["lr"], weight_decay = opt["weight_decay"])
        
        train_acc = []
        val_acc = []
        test_acc = []

        train_l = []
        val_l = []
        
        for epoch in range(opt['epochs']):
            #start_time = time.time()
            loss = train_func(model, optimizer, data)
            train_l.append(loss)

            tmp_train_acc, tmp_val_acc, tmp_test_acc, tmp_val_loss = test_func(model, data, opt)

            train_acc.append(tmp_train_acc)
            val_acc.append(tmp_val_acc)
            test_acc.append(tmp_test_acc)
            val_l.append(tmp_val_loss)

            #print(f'Rep: {rep} Epoch: {epoch}, train_loss:{loss:.4f}, '
            #    f'train_acc: {tmp_train_acc:.4f}, val_acc:{tmp_val_acc:.4f}, test_acc: {tmp_test_acc:.4f}')
            
        argmax = np.argmax(val_acc)
        #print(f"Max test acc is {test_acc[argmax]} and val acc is {val_acc[argmax]} at epoch {argmax}")
        best_val_acc.append(val_acc[argmax])
        best_test_acc.append(test_acc[argmax])
    
    #print('---')
    #for rep in range(10):
    #    print(f'Rep:{rep}, max val acc:{best_val_acc[rep]:.4f}, max test acc:{best_test_acc[rep]:.4f}')
    #print('---')
    #print(f'mean val acc: {np.mean(best_val_acc):.4f} std val acc:{np.std(best_val_acc)}, '
    #      f'mean test acc :{np.mean(best_test_acc):.4f}, std test acc:{np.std(best_test_acc)}')

    wandb.log({"val_acc": np.mean(best_val_acc), "test_acc": np.mean(best_test_acc), "test_acc_std": np.std(best_test_acc)})
        
        
    


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()

    #Training settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

    parser.add_argument('--num_h_layers', type=int, default=0, help ='LAup,LAdown,ActUp, ActDown')
    parser.add_argument('--hidden', type=int, default=16, help='Hidden dimension.')
    parser.add_argument('--input_dropout', type = float, default=0.5, help='Dropout for input.')
    parser.add_argument('--dataset', type=str, default='Cora', help='Cora,Citeseer,Pubmed,Film...' )
    
    parser.add_argument('--geom_gcn_splits', type=str, default='True', help='Use geom-gcn split')
    parser.add_argument('--rewiring', type=str, default=None, help='Graph rewiring.')

    args = parser.parse_args()

    #opt = vars(args)
    #wandb.init(project='your_project_name', entity='alanjohnvarghese')

    
    sweep_config = {
        'method': 'random',  # or 'bayes', 'grid', 'random'
        'metric': {
        'name': 'val_acc',
        'goal': 'maximize'   
        },
        'parameters': {
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 5e-4,
                'max': 1e-2
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 5e-2
            },
            'num_h_layers': {
                'values': [1, 2, 3, 4, 5]
            },
            'hidden': {
                'values': [8, 16, 32, 64, 128]
            },
            'input_dropout': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.6
            },
            'seed':{
                'value':args.seed
            },
            'epochs':{
                'value':args.epochs
            },
            'dataset':{
                'value':args.dataset
            },
            'geom_gcn_splits':{
                'value':args.geom_gcn_splits
            },
            'rewiring':{
                'value':args.rewiring
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project= args.dataset + '_source', entity='alanjohnvarghese')
    wandb.agent(sweep_id, function = train_sgnn, count=500)
