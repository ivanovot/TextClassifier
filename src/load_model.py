import torch
import yaml

from .classifier import Classifier
from .bert import Bert

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = yaml.safe_load(open('config.yaml', 'r'))

model = Classifier(Bert(config['model']['bert_name']))

model.load_state_dict(torch.load(config['predct']['use_model'], map_location=torch.device(device), weights_only=True))

model = model.to(device)

model.eval()