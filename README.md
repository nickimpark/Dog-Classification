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
