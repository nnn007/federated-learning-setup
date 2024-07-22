import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import requests
import pickle
import io
# import threading
from torch.utils.data import Subset
import numpy as np


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 10)

    def forward(self, x):
        return self.model(x)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_indices = np.random.choice(len(train_dataset), size=5000, replace=False)
train_subset = Subset(train_dataset, train_indices)
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=16, shuffle=True)


def download_model():
    response = requests.get("http://127.0.0.1:8000/download_model/")
    global_model_data = response.content
    global_model = MobileNet()
    global_model.load_state_dict(torch.load(io.BytesIO(global_model_data)))
    return global_model


def train_local(model, dataloader):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(1):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()


def upload_weights(local_weights):
    serialized_weights = pickle.dumps(local_weights)
    files = {'file': ('weights.pkl', serialized_weights)}
    response = requests.post("http://127.0.0.1:8000/upload_weights/", files=files)
    print(response.json())
