import os
import urllib.request
import tarfile
from config import DATA_PATH, TRAIN_PATH, VAL_PATH, DATA_URL

os.makedirs(DATA_PATH, exist_ok=True)
print('Loading data...')
urllib.request.urlretrieve(DATA_URL, os.path.join(DATA_PATH, 'imagewoof2-320.tgz'))
print('Success. Unpacking...')
file = tarfile.open(os.path.join(DATA_PATH, 'imagewoof2-320.tgz'))
file.extractall('./')
file.close()
os.remove(os.path.join(DATA_PATH, 'imagewoof2-320.tgz'))
os.rmdir(DATA_PATH)
os.rename('imagewoof2-320', DATA_PATH)
print('Success. All data collected.')

label_dict = {
    'n02099601': 'Golden retriever',
    'n02086240': 'Shih-Tzu',
    'n02087394': 'Rhodesian ridgeback',
    'n02105641': 'Old English sheepdog',
    'n02088364': 'Beagle',
    'n02111889': 'Samoyed',
    'n02093754': 'Border terrier',
    'n02089973': 'English foxhound',
    'n02096294': 'Australian terrier',
    'n02115641': 'Dingo'
}

for k in label_dict.keys():
    os.rename(os.path.join(TRAIN_PATH, k), os.path.join(TRAIN_PATH, label_dict[k]))
    os.rename(os.path.join(VAL_PATH, k), os.path.join(VAL_PATH, label_dict[k]))

print('Class names transformed successfully.')
