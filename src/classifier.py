import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.head = nn.Linear(self.bert.model.config.hidden_size, 1)
        
    def forward(self, texts:list[str]):
        embeddings = self.bert(texts)
        return torch.sigmoid(self.head(embeddings)).squeeze(1)