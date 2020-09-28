import pickle
import torch
from torch import nn
from torch.utils.data import Dataset

def data_load(filename):
    with open(filename, 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_val = pickle.load(inp)
        y_val = pickle.load(inp)
    return word2id, id2word, tag2id, id2tag, x_train, y_train, x_test, y_test, x_val, y_val

class NERDataset(Dataset):
    def __init__(self, x, y, *args, **kwargs):
        self.data = [{'x':x[i],'y':y[i]} for i in range(x.shape[0])]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)