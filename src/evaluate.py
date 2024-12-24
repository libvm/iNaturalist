import yaml
import os
from dvclive import Live
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dir = "data_split/test"


def get_test_loader():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return test_loader


def get_model():
    model = models.resnet50(
        pretrained=True,
    )

    num_classes = len(os.listdir(test_dir))
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)

    model.load_state_dict(torch.load("models/model.pth"))
    model.eval()

    return model


def evaluate_model():
    model = get_model()
    test_loader = get_test_loader()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    with Live(dir="dvclive_acc", dvcyaml=False) as live:
        live.log_metric("accuracy", accuracy)


if __name__ == "__main__":
    evaluate_model()
