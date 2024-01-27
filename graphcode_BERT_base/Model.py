import numpy as np
import torch.nn.functional as f
import torch.nn as nn
import torch
from transformers import AutoConfig, AutoModel

class _20SampleModel(nn.Module): 
    def __init__(self, model_path):
        super(_20SampleModel, self).__init__()
        
        config = AutoConfig.from_pretrained(model_path)
        
        self.model = AutoModel.from_pretrained(model_path)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.top = nn.Linear(config.hidden_size+1, 1) # for train_fts

    def forward(self, ids, mask, fts, labels=None):
        x = self.model(ids, mask)[0]
        x = torch.cat((x[:, 0, :], fts), 1)
        
        x1 = self.top(self.dropout1(x))
        x2 = self.top(self.dropout2(x))
        x3 = self.top(self.dropout3(x))
        x4 = self.top(self.dropout4(x))
        x5 = self.top(self.dropout5(x))
        x = (x1 + x2 + x3 + x4 + x5) / 5
        return x

class PairwiseModel(nn.Module):
    def __init__(self, model_path):
        super(PairwiseModel, self).__init__()

        config = AutoConfig.from_pretrained(model_path)
        
        self.model = AutoModel.from_pretrained(model_path)
        self.top = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, ids, mask):
        x = self.model(ids, mask)[0]
        x = self.dropout(x)
        x = self.top(x[:, 0, :])
        x = torch.sigmoid(x) 
        return x

