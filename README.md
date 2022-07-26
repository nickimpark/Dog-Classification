# Dog-Classification

## Demo

Microservice at: https://dog-classification-nickimpark.herokuapp.com/

## About

This project is aimed at creating a microservice for classifying dogs by breeds. The EfficientNet-B0 Model was trained on ImageWoof2-320 Dataset and has 74.42% validation accuracy.

Project repository contains:
* config.py - congiguration file
* flask-app.py - flask app
* load_data.py - load ImageWoof data if you want to train your own model
* model_eval.py - script for model evaluating (Validation accuracy: 74.42%)
* model_inference.py - simple inference of trained model (from ./input)
* telegram_bot.py - script for telegram bot (add your token and run)
* train.py - model training script, model will be available at ./models/

Stack: Flask, PyTorch

Don't forget about **requirements.txt** file:
```
pip install -r requirements.txt
```
or
```
pip install -r requirements_cuda.txt -f https://download.pytorch.org/whl/torch_stable.html
```
if you want to use CUDA.

## Model Performance

Model metrics (from model_eval.py):
* **Validation accuracy: 74.42%**
* Precision for class Australian terrier: 73.22%
* Precision for class Beagle: 63.40%
* Precision for class Border terrier: 77.31%
* Precision for class Dingo: 75.61%
* Precision for class English foxhound: 50.89%
* Precision for class Golden retriever: 75.56%
* Precision for class Old English sheepdog: 79.15%
* Precision for class Rhodesian ridgeback: 79.90%
* Precision for class Samoyed: 87.65%

The worst performance is on class English foxhound. One of the reasons for this could be the fact that there were the least number of images of this class in the training set. Also, the cause of errors can be the presence in the training and validation sets of images in which, in addition to the dog, there are many extra objects.

The metrics could have been better if a more complex model had been chosen, but this requires significant computational resources. **If you test the model on images that clearly show the dog, you will rarely see incorrect predictions.**


## QuickStart with Docker

Clone repository:
```
git clone https://github.com/nickimpark/Dog-Classification
```
Build docker image:
```
docker build -t dog-classification .
```
Run docker:
```
docker run dog-classification
```
