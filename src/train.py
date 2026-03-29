import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from collections import Counter
from model import EmotionResNet
from dataloader import get_dataloader
from evaluate import evaluate, print_classification_report
from utils import save_model


def train(data_dir, epochs=20, lr = 3e-5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_loader, test_loader, classes = get_dataloader(data_dir)

    model = EmotionResNet(num_classes=len(classes)).to(device)

    labels = [label for _, label in train_loader.dataset.samples]

    class_counts = Counter(labels)

    total = sum(class_counts.values())
    weights = [total / class_counts[i] for i in range(len(class_counts))]

    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_accuracies = []

    best_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            loop.set_postfix(loss=loss.item())

        acc = evaluate(model, test_loader, device)

        epoch_loss = running_loss / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        val_accuracies.append(acc)

        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f} | Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            save_model(model, "models/emotion_resnet.pth")
            print("🔥 Best model saved")

    print(f"\nTraining complete. Best Accuracy: {best_acc:.2f}%")

    acc, all_preds, all_labels = evaluate(
        model, test_loader, device, classes, return_preds=True
    )

    print(f"\nFinal Accuracy: {acc:.2f}%")
    print_classification_report(all_labels, all_preds, classes)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("resnet_training_plot.png")
    plt.show()


if __name__ == "__main__":
    train("data")