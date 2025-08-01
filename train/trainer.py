import torch.optim as optim
import torch.nn.functional as F
import torch

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0

    for inputs, labels in dataloader:  # Fixed: inputs first, then labels
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in dataloader:  # Fixed: inputs first, then labels
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy



