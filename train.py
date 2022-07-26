from config import DATA_PATH, TRAIN_PATH, VAL_PATH, SAVE_PATH

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from efficientnet_pytorch import EfficientNet
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.RandomOrder([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=5, scale=(0.8, 1.2), fillcolor=None),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05),
            transforms.GaussianBlur(kernel_size=3)
        ]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # from ImageNet
    ])

val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # from ImageNet
    ])

train_data = datasets.ImageFolder(TRAIN_PATH, train_transform)

val_data = datasets.ImageFolder(VAL_PATH, val_transform)

image_data = {'train': train_data, 'val': val_data}

dataloader = {x: torch.utils.data.DataLoader(image_data[x], batch_size=16,
                                             shuffle=True, num_workers=2, pin_memory=True)
              for x in ['train', 'val']}

data_size = {x: len(image_data[x]) for x in ['train', 'val']}

class_name = train_data.classes

num_classes = len(class_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print(f'Device type: {device.type}')


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_log = open('train_log.txt', 'w')
    train_log.close()
    train_log = open('train_log.txt', 'a')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        train_log.write(f'Epoch {epoch + 1}/{num_epochs}\n')
        train_log.write('-' * 10)
        train_log.write('\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward & optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # history
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / data_size[phase]
            if phase == 'train':
                epoch_train_loss = epoch_loss
            if phase == 'val':
                epoch_val_loss = epoch_loss
            epoch_acc = running_corrects.double() / data_size[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            train_log.write(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        # early stopping
        early_stopping(epoch_train_loss, epoch_val_loss)
        if early_stopping.early_stop:
            print('Early stopping...')
            train_log.write('Early stopping...')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    train_log.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
    train_log.write(f'Best val Acc: {best_acc:4f}\n')
    train_log.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# All parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Decay LR by a factor of 0.2 every 25 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.2)

# Early stopping
early_stopping = EarlyStopping(tolerance=5, min_delta=1)

if __name__ == '__main__':
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=100)
    os.makedirs(SAVE_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'model.state'))
    print('The best model saved successfully')
