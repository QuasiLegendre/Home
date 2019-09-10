from torch_geometric.data import Data, DataLoader
from net import Net
import numpy as np
import torch
import torch.nn.functional as F
from pandas import read_csv
from utils import ReadThemAll
path = './Graphs/JHU/'
data_groups = ReadThemAll(path)
results = []
labels_init = read_csv('HNU1R.csv').sort_values(by=['SUBID', 'SESSION']).get('SEX').values.tolist()
labels_all = [[i-1] for i in labels_init]
#for i in labels_init:
#    if i == 1:
#        labels_all.append([1, 0])
#    else:
#        labels_all.append([0, 1])
keys = sorted(data_groups.keys())
labels_dict = {}
init_num = 0
for k in keys:
    labels_dict[k] = labels_all[init_num:init_num+len(data_groups[k])]
    init_num += len(data_groups[k])
print(init_num)
#keys_test = [keys[i] for i in range(0, 30, 3)]
step_num = 3
for k in range(0, 30, 3):
    test_keys = keys[k:k+step_num]
    train_keys = keys[:k] + keys[k+step_num:]
    dataset_train = []
    labels_train = []
    dataset_test = []
    labels_test = []
    for i in train_keys:
        dataset_train.extend(data_groups[i])
        labels_train.extend(labels_dict[i])
    for i in test_keys:
        dataset_test.extend(data_groups[i])
        labels_test.extend(labels_dict[i])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    dataset = [k.to(device) for k in dataset_train]
    labels = torch.tensor(labels_train, dtype=torch.long).to(device)
    dataset_t = [k.to(device) for k in dataset_test]
    labels_t = torch.tensor(labels_test, dtype=torch.long).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(1):
        label_num = 0
        for data in dataset:
            optimizer.zero_grad()
            out = model(data)
            #print(out)
            loss = F.cross_entropy(out, labels[label_num])
            loss.backward()
            optimizer.step()
            label_num += 1
    model.eval()
    ######
    res = 0.0
    for data, label in zip(dataset_t, labels_t):
        _, pred = model(data).max(dim=1)
        res += float(pred.eq(label).item())
    print(res)
    res /= len(labels_test)
    results.append(res)
print(results)
print(sum(results)/len(results))
torch.cuda.empty_cache()