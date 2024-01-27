import torch
from transformers import AutoModel
import torch.nn as nn

class MarkdownModel(nn.Module):
    def __init__(self, model_name_or_path):
        super(MarkdownModel, self).__init__()
        
        self.distill_bert = (AutoModel.from_pretrained(model_name_or_path))
        
        self.top = nn.Linear(512, 1)

        self.dropout = nn.Dropout(0.2)
        
    def forward(self, ids, mask):
        x = self.distill_bert(ids, mask)[0]
        x = self.dropout(x)
        x = self.top(x[:, 0, :])
        x = torch.sigmoid(x) 
        return x