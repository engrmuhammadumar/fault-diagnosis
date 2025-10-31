import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_model(model, train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    writer = SummaryWriter('runs/pump_diagnosis')
    
    best_val_acc = 0
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}'):
            time_series = batch['time_series'].to(device)
            scalogram = batch['scalogram'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(time_series, scalogram)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                time_series = batch['time_series'].to(device)
                scalogram = batch['scalogram'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(time_series, scalogram)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss/len(val_loader), epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    writer.close()
    return model

def evaluate_model(model, test_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    test_correct, test_total = 0, 0
    all_predictions, all_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            time_series = batch['time_series'].to(device)
            scalogram = batch['scalogram'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(time_series, scalogram)
            _, predicted = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    return test_acc, all_predictions, all_labels

