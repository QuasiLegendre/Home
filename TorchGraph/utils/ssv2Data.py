import numpy as np
import torch
from torch_geometric.data import Data
def ssv2Data(file_name, undirected=True):
    text_file = open(file_name, 'r')
    graph_list = list(
        map(lambda x: tuple(*map(lambda y: [int(y[0]), int(y[1]), float(y[2])], [x.split('\n')[0].split(' ')])),
            text_file.readlines()))
    text_file.close()
    if undirected:
        X = [1]*len(graph_list)
        edge_index = [[], []]
        edge_attr = []
        for (x, y, w) in graph_list:
            #X.append(x)
            #X.append(y)
            edge_index[0].extend([x, y])
            edge_index[1].extend([y, x])
            edge_attr.extend([w, w])
        #X = np.unique(X)
        #X.sort()
        X = np.asarray(X)
        X = torch.tensor(X.reshape([len(X), -1]), dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = np.asarray(edge_attr)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)#.reshape([len(edge_attr), -1]))
        return Data(x=X, edge_index=edge_index, edge_attr=edge_attr)
