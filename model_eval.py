from config import DATA_PATH, TRAIN_PATH, VAL_PATH, SAVE_PATH

import torch
import torchvision
from torchvision import datasets, models, transforms
from efficientnet_pytorch import EfficientNet
import os

val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # from ImageNet
    ])

val_data = datasets.ImageFolder(VAL_PATH, val_transform)

val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=8,
                                             shuffle=True, num_workers=2, pin_memory=True)

data_size = len(val_data)

class_name = val_data.classes

num_classes = len(class_name)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluate_model(model):
    model.eval()   # Set model to evaluate mode
    running_corrects = 0
    for inputs, labels in val_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / data_size
    print(f'Validation accuracy: {100 * acc:.2f}%')


model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
model.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'model.state')))
model = model.to(device)

if __name__ == '__main__':
    evaluate_model(model)
