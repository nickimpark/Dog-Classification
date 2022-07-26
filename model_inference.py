from config import SAVE_PATH, INPUT_PATH

import torch
from torch import nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import os
from PIL import Image
import argparse

input_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # from ImageNet
])

idx_to_class = {0: 'Australian terrier',
                1: 'Beagle',
                2: 'Border terrier',
                3: 'Dingo',
                4: 'English foxhound',
                5: 'Golden retriever',
                6: 'Old English sheepdog',
                7: 'Rhodesian ridgeback',
                8: 'Samoyed',
                9: 'Shih-Tzu'
                }

model = EfficientNet.from_name('efficientnet-b0', num_classes=10)
model.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'model.state')))
model.eval()


def predict_all():
    for file in os.listdir(INPUT_PATH):
        print(f'Image name: {file}')
        img = Image.open(os.path.join(INPUT_PATH, file))
        x = input_transform(img)
        x.unsqueeze_(0)
        with torch.set_grad_enabled(False):
            output = model(x)
            _, pred = torch.max(output, 1)
            probs = nn.Softmax(-1).forward(output)

        pred = int(pred)
        pred_class = idx_to_class[pred]
        proba = float(probs.numpy()[0, pred])

        print(f'Predicted class: {pred_class}, Probability: {round(100 * proba, 2)}%')


def predict_one(filename):
    img = Image.open(os.path.join(INPUT_PATH, filename))
    x = input_transform(img)
    x.unsqueeze_(0)
    with torch.set_grad_enabled(False):
        output = model(x)
        _, pred = torch.max(output, 1)
        probs = nn.Softmax(-1).forward(output)

    pred = int(pred)
    pred_class = idx_to_class[pred]
    proba = float(probs.numpy()[0, pred])

    print(f'Predicted class: {pred_class}, Probability: {round(100 * proba, 2)}%')


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None, help='Image name from the input folder for classification')
    return parser.parse_args()


args = args_parse()
if args.image is not None:
    predict_one(args.image)
else:
    predict_all()
