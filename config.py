import os

DATA_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz'
DATA_PATH = 'data'
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
VAL_PATH = os.path.join(DATA_PATH, 'val')
SAVE_PATH = 'models'
INPUT_PATH = 'input'
