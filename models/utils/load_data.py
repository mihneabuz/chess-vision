import cv2
import json
from os import listdir, path
from random import sample
from tqdm import tqdm

DATA = 'boards/data'

def download_data():
    print('downloading dataset...', end='', flush=True)
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('thefamousrat/synthetic-chess-board-images', path='boards', unzip=True)
    print('done')

def load_data(max=-1):
    if not path.isdir(DATA):
        download_data()

    files = [file[:-4] for file in listdir(DATA) if file.endswith('.jpg')]

    if (max > 0):
        files = sample(files, max)

    for file in tqdm(files, desc='loading images'):
        image = cv2.imread(DATA + '/' + file + '.jpg')
        annotations = json.load(open(DATA + '/' + file + '.json'))
        yield image, annotations


if __name__ == "__main__":
    for image, labels in load_data(1):
        print(image.shape)
        print(labels)
