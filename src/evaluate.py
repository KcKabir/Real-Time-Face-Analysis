import torch
from sklearn.metrics import classification_report

def evaluate(model, loader, device, classes=None, return_preds=False):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []
     
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            if return_preds:
                all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = 100 * correct / total
    if return_preds:
        return acc, all_preds, all_labels

    return acc

def print_classification_report(all_labels, all_preds, classes):
    report = classification_report(
        all_labels,
        all_preds,
        target_names=classes
    )

    print("\n📋 Classification Report:\n")
    print(report)