import torch
from sklearn.metrics import classification_report

def evaluate_model(model, loader, device, class_names):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(y.numpy())

    print(classification_report(trues, preds, target_names=class_names))
