from config import SAVE_PATH

import os
import uuid
import urllib
from flask import Flask, render_template, request, send_file
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

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

app = Flask(__name__)

ALLOWED_EXT = {'jpg', 'jpeg', 'png'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


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


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images')
    if request.method == 'POST':
        if request.form:
            link = request.form.get('link')
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path, "wb")
                output.write(resource.read())
                output.close()
                img = filename
                pred_class, proba = predict_one(img_path)
                proba = round(100 * proba, 2)

                predictions = {
                    "class": pred_class,
                    "prob": proba
                }

            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accessible or inappropriate input'

            if len(error) == 0:
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)
        elif request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename
                pred_class, proba = predict_one(img_path)
                proba = round(100 * proba, 2)

                predictions = {
                    "class": pred_class,
                    "prob": proba
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if len(error) == 0:
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
