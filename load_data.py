from cv2 import cv2
import json
from os import listdir, path
from random import sample
from tqdm import tqdm

from process import crop_board, crop_pieces

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

def load_board_images(max=-1):
    for image, annotations in load_data(max=max):
        yield image, annotations['corners']

def load_piece_images(max=-1):
    for image, annotations in load_data(max=max):
        board_image = crop_board(image, annotations['corners'])
        pieces, labels = crop_pieces(board_image, annotations['config'])
        yield from zip(pieces, labels)

if __name__ == "__main__":
    for image, labels in load_data(1):
        print(image.shape)
        print(labels)

    for image, labels in load_board_images(1):
        print(image.shape)
        print(labels)

    for piece, label in load_piece_images(1):
        print(piece.shape)
        print(label)
        break
