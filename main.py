import torch
import torch.nn as nn
from torch import optim
from train.trainer import train, evaluate
from models.encoder import TransformerClassifier
from utils.data import build_dataset
from utils.residual import NormType
from config import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, vocab = build_dataset(config["batch_size"], config["max_len"])
norm_type = NormType.DYT if config["use_dyt"] else NormType.LAYER_NORM

model = TransformerClassifier(
    vocab_size=vocab,
    config=config
).to(device)

optimizer = optim.Adam(model.parameters(), lr=config["lr"])
criterion = nn.CrossEntropyLoss()
for epoch in range(config["epochs"]):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}/{config['epochs']} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
