import torch
from sklearn.metrics import accuracy_score

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(y.numpy())

        val_acc = accuracy_score(trues, preds)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    return best_acc
