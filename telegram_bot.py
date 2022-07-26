# -*- coding: utf-8 -*-
from config import SAVE_PATH

import os
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import telebot
from PIL import Image

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
model.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'model.state'), map_location='cpu'))
model.eval()


def predict_one(filename):
    img = Image.open(filename)
    x = input_transform(img)
    x.unsqueeze_(0)
    with torch.set_grad_enabled(False):
        output = model(x)
        _, pred = torch.max(output, 1)
        probs = nn.Softmax(-1).forward(output)

    pred = int(pred)
    pred_class = idx_to_class[pred]
    proba = float(probs.numpy()[0, pred])
    return pred_class, proba


bot = telebot.TeleBot('your_token')  # Add your token!


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.send_message(message.from_user.id,
                     'Здравствуйте. Данный бот предназначен для классификации собак по нескольким породам: Australian terrier, Beagle, Border terrier, Dingo, English foxhound, Golden retriever, Old English sheepdog, Rhodesian ridgeback, Samoyed, Shih-Tzu.')
    bot.send_message(message.from_user.id,
                     'Отправьте фотографию собаки, чтобы узнать ее породу.')


@bot.message_handler(content_types=['photo'])
def get_photo_messages(message):
    fileID = message.photo[-1].file_id
    print('fileID =', fileID)
    file_info = bot.get_file(fileID)
    print('file.file_path =', file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('input/tele_image.jpg', 'wb') as new_file:
        new_file.write(downloaded_file)
    pred_class, proba = predict_one('input/tele_image.jpg')
    proba = round(100 * proba, 2)
    bot.send_message(message.from_user.id, f'С вероятностью {proba}% собака на картинке относится к породе {pred_class}.')


@bot.message_handler(content_types=['text'])  # Message handler (method)
def get_text_messages(message):
    if message.text == 'Test':
        bot.send_message(message.from_user.id, 'Test')
    else:
        bot.send_message(message.from_user.id, 'Отправьте фотографию собаки, чтобы узнать ее породу.')


# Polling
bot.polling(none_stop=True, interval=0)
