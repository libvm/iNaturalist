import torch
import torch.nn as nn
import torch.optim as optim
from dvclive import Live
import yaml
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = "data_split/train"
val_dir = "data_split/val"


def get_loaders():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Используем ImageFolder для загрузки данных
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    # Создаем DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


def get_model():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]
    model = models.resnet50(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    num_classes = len(os.listdir(train_dir))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    num_epochs = params["epochs"]
    return model, criterion, optimizer, num_epochs


def train_model():
    model, criterion, optimizer, num_epochs = get_model()
    train_loader, val_loader = get_loaders()

    with Live(dir="dvclive_loss", dvcyaml=False) as live:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss / len(train_loader)

            live.log_metric("train_loss", train_loss)

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = correct / total

            live.log_metric("val_loss", val_loss)
            live.log_metric("val_acc", val_acc)
            live.next_step()

        # Сохранение модели
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/model.pth")


if __name__ == "__main__":
    train_model()
