from fastapi import FastAPI, UploadFile, File, Response
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import pickle
from torch.utils.data import Subset
import numpy as np
import uvicorn

app = FastAPI()


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 10)

    def forward(self, x):
        return self.model(x)


global_model = MobileNet()
best_model = MobileNet()
best_accuracy = 0

# Load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_indices = np.random.choice(len(test_dataset), size=1000, replace=False)
test_subset = Subset(test_dataset, test_indices)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=16, shuffle=False)


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(filename):
    model = MobileNet()
    model.load_state_dict(torch.load(filename))
    return model


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


performance_data = []

save_model(global_model, 'global_model.pth')
save_model(best_model, 'best_model.pth')


@app.post("/upload_weights/")
async def upload_weights(file: UploadFile = File(...)):
    global best_accuracy
    contents = await file.read()
    local_weights = pickle.loads(contents)

    global_state_dict = global_model.state_dict()
    for key in global_state_dict.keys():
        global_state_dict[key] = global_state_dict[key].float() + local_weights[key].float()
        global_state_dict[key] /= 2
    global_model.load_state_dict(global_state_dict)

    accuracy = evaluate(global_model, test_loader)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model.load_state_dict(global_model.state_dict())
        save_model(best_model, 'best_model.pth')

    performance_data.append({"round": len(performance_data) + 1, "accuracy": accuracy})
    return {"accuracy": accuracy}


@app.get("/download_model/")
async def download_model():
    with open('best_model.pth', 'rb') as f:
        model_data = f.read()
    return Response(model_data, media_type='application/octet-stream')


@app.get("/performance/")
async def get_performance():
    return performance_data


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
