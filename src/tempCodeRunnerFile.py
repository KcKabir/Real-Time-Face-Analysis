import torch
from torch import nn
from tqdm import tqdm
import os

from model import Emotion_CNN_Model
from dataloader import get_dataloader
from evaluate import evaluate
from utils import save_model

def train_step(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Your device is {device}")
    
    train_loader, test_loader, classes = get_dataloader(data_dir)
    model = Emotion_CNN_Model(num_classes=len(classes)).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_acc = 0.0
    os.makedirs("models", exist_ok=True)
    
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        acc = evaluate(model, test_loader, device)
        
        print(f"[Epoch {epoch+1}] Loss: {running_loss:.4f} | Val Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            save_model(model, "models/emotion_cnn.pth")
            print("🔥 Best model saved")
        
    print(f"Training has been completed. Best Accuracy is {best_acc:.2f}%")
    
if __name__ == "__main__":
    train_step("data")
    
    