import os.path
import pickle

from typing import Dict

import torch
from torch.autograd import Variable
from torch_geometric.nn import to_hetero, to_hetero_with_bases
from torch_geometric.data import HeteroData

import data_loader_compact
from models import GCN, GCN_SimpleMultipleOutput, GNN_MultipleOutput, GAT, GNN_Het, GNN_Het_EA, VerStateModel, ARMANet, \
    GATNet, ARMA_Het, StateModelEncoder
from random import shuffle
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

BALANCE_DATASET = False


class PredictStateVectorHetGNN:
    """predicts ExpectedStateNumber using Heterogeneous GNN"""

    def __init__(self):
        self.state_maps = {}
        self.start()

    def start(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        # dataset = data_loader_compact.get_data_hetero_vector()
        dataset = []
        dataset_dir = "./dataset_t/"
        #dataset_dir = "./dataset_t_s/"
        for file in os.listdir(dataset_dir):
            print(file)
            with open(os.path.join(dataset_dir, file), "rb") as f:
                dat = (pickle.load(f))
                if BALANCE_DATASET and len(dat) > 5000:
                    dat = dat[:5000]
                    print("Part of the dataset is chosen!")
                dataset.extend(dat)

        torch.manual_seed(12345)
        # print(dataset[3].y[:, 3])
        shuffle(dataset)

        split_at = round(len(dataset) * 0.85)

        train_dataset = dataset[:split_at]
        test_dataset = dataset[split_at:]

        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of test graphs: {len(test_dataset)}')

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)  # TODO: add learning by batches!
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = StateModelEncoder(dataset[0].metadata(), hidden_channels=64, out_channels=8)
        # model = GNN_Het(hidden_channels=64, out_channels=1).to(device)
        #model = ARMA_Het(hidden_channels=64, out_channels=1)
        '''model = ARMANet(
            hidden_channels=64,
            num_classes=1,
            num_stacks=1,
            num_layers=1).to(device)'''
        '''model = GATNet(in_channels=6,
                        hidden_channels=64,
                        num_classes=1).to(device)'''

        # print(dataset[0].__dict__)
        #model = to_hetero(model, dataset[0].metadata(), aggr='sum').to(device)
        # model = to_hetero_with_bases(model, dataset[0].metadata(), num_bases=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, 21):
            self.train(model, train_loader, optimizer)
            train_acc = self.tst(model, train_loader)
            test_acc = self.tst(model, test_loader)
            # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            print(f'Epoch: {epoch:03d}, Train Loss: {train_acc:.6f}, Test Loss: {test_acc:.6f}')

        self.save(model, "./saved_models")

    # loss function from link prediction example
    def weighted_mse_loss(self, pred, target, weight=None):
        weight = 1. if weight is None else weight[target].to(pred.dtype)
        return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()

    def train(self, model, train_loader, optimizer):
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            # print(data)
            out = model(data.x_dict, data.edge_index_dict)
            pred = out['state_vertex']
            target = data.y
            #target = torch.reshape(data.y[:, 4], pred.shape)
            loss = F.mse_loss(pred, target)
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def tst(self, model, loader):
        model.eval()
        for data in loader:
            out = model(data.x_dict, data.edge_index_dict)
            pred = out['state_vertex']
            target = data.y
            #target = torch.reshape(data.y[:, 4], pred.shape)
            loss = F.mse_loss(pred, target)
            # print(pred, target)
            # correct += int(pred == target)
        return loss  # correct / len(loader.dataset)

    @staticmethod
    def predict_state(model, data: HeteroData, state_map: Dict[int, int]) -> int:
        '''Gets state id from model and heterogeneous graph
        data.state_map - maps real state id to state index '''
        state_map = {v: k for k, v in state_map.items()}  # inversion for prediction
        out = model(data.x_dict, data.edge_index_dict)
        return state_map[int(out['state_vertex'].argmax(dim=0)[0])]

    def save(self, model, dir):
        filepath = os.path.join(dir, "GNN_state_pred_het_dict")
        # case 1
        torch.save(model.state_dict(), filepath)
        # case 2
        filepath = os.path.join(dir, "GNN_state_pred_het_full")
        torch.save(model, filepath)


if __name__ == '__main__':
    PredictStateVectorHetGNN()
