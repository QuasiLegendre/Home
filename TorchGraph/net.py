import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)
        #self.linear = nn.Linear(edge_num*2, 1)
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        #x = x.view(-1)
        #x = self.linear(x)
        x = torch.sum(x, dim=0)
        #print(x)
        x = x.unsqueeze(0)

        return F.softmax(x, dim=1)