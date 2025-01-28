import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class Bert(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def forward(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.model.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]