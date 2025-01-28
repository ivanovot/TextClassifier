from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label
    
if __name__ == "__main__":
    import pandas as pd
    import torch
    
    splits = {'train': 'train.jsonl', 'test': 'test.jsonl'}
    
    df_train = pd.read_json("hf://datasets/AlexSham/Toxic_Russian_Comments/" + splits["train"], lines=True)
    df_test = pd.read_json("hf://datasets/AlexSham/Toxic_Russian_Comments/" + splits["test"], lines=True)
    
    dataset_train = TextDataset(df_train)
    dataset_test = TextDataset(df_test)
    
    torch.save(dataset_train, 'data/dataset_train.pt')
    torch.save(dataset_test, 'data/dataset_test.pt')