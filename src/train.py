import logging
import yaml
import torch
import time
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd

from .classifier import Classifier
from .bert import Bert
from .dataset import TextDataset

def setup_logging(config):
    logging.basicConfig(
        filename=os.path.join(config['logging']['log_dir'], "log.log"),
        filemode='w',
        level=config['logging']['level'],
        format=config['logging']['format']
    )
    return logging.getLogger(__name__)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for texts, labels in dataloader:
            labels = labels.float().to(device)
            outputs = model(texts).squeeze()
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += ((outputs >= 0.5).float() == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / total, correct / total

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))
    logger = setup_logging(config)
    
    logger.info("Starting training process")
    logger.info(f"Configuration: {config}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Инициализация модели
    model = Classifier(Bert(config['model']['bert_name'])).to(device)
    logger.info("Model initialized")
    
    # Загрузка данных
    train_dataset = torch.load(config['data']['train_path'])
    test_dataset = torch.load(config['data']['test_path'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config['data']['batch_size']),
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config['data']['batch_size']),
        shuffle=False
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Оптимизатор
    optimizer = optim.Adam(
        model.parameters(), 
        lr=float(config['training']['learning_rate'])
    )
    criterion = nn.BCELoss()

    # Для записи результатов обучения
    results = []
    
    for epoch in range(int(config['training']['epochs'])):
        start_time = time.time()
        
        # Обучение
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100):
            labels = labels.float().to(device)
            optimizer.zero_grad()
            
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            train_correct += ((outputs >= 0.5).float() == labels).sum().item()
        
        # Оценка
        train_loss /= len(train_dataset)
        train_acc = train_correct / len(train_dataset)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        # Сохранение результатов
        results.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc
        })
        
        # Логирование
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1} [{epoch_time:.1f}s]")
        logger.info(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        logger.info(f"Test  Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
        
        torch.save(model.state_dict(), 
                      os.path.join(config['training']['save_dir'], f"model_{epoch+1}.pth"))
    
    # Сохранение результатов в CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(config['logging']['log_dir'], "training_results.csv"), index=False)
    
    # Финализация обучения
    logger.info("Training completed")
