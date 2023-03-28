import cv2
import json
from os import listdir, path
from random import sample
from tqdm import tqdm
from copy import deepcopy

# this is stupid but whatever
rotate_map = {
    "A1": "H1", "A2": "G1", "A3": "F1", "A4": "E1", "A5": "D1", "A6": "C1", "A7": "B1", "A8": "A1",
    "B1": "H2", "B2": "G2", "B3": "F2", "B4": "E2", "B5": "D2", "B6": "C2", "B7": "B2", "B8": "A2",
    "C1": "H3", "C2": "G3", "C3": "F3", "C4": "E3", "C5": "D3", "C6": "C3", "C7": "B3", "C8": "A3",
    "D1": "H4", "D2": "G4", "D3": "F4", "D4": "E4", "D5": "D4", "D6": "C4", "D7": "B4", "D8": "A4",
    "E1": "H5", "E2": "G5", "E3": "F5", "E4": "E5", "E5": "D5", "E6": "C5", "E7": "B5", "E8": "A5",
    "F1": "H6", "F2": "G6", "F3": "F6", "F4": "E6", "F5": "D6", "F6": "C6", "F7": "B6", "F8": "A6",
    "G1": "H7", "G2": "G7", "G3": "F7", "G4": "E7", "G5": "D7", "G6": "C7", "G7": "B7", "G8": "A7",
    "H1": "H8", "H2": "G8", "H3": "F8", "H4": "E8", "H5": "D8", "H6": "C8", "H7": "B8", "H8": "A8"
}

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
        annotations = sort_corners(json.load(open(DATA + '/' + file + '.json')))
        yield image, annotations


def sort_corners(annotations):
    def cost(c):
        return \
            (c[0][0] + c[0][1]) + \
            (c[1][0] + c[1][1]) * 2 + \
            (c[2][0] + c[2][1]) * 4 + \
            (c[3][0] + c[3][1]) * 8

    def shift(annotations):
        corners = annotations['corners']
        config = annotations['config']
        config = { rotate_map[key]: val for key, val in config.items() }
        return {
            'corners': [corners[1], corners[3], corners[0], corners[2]],
            'config': config
        }

    best = deepcopy(annotations)
    best_cost = cost(best['corners'])
    for _ in range(4):
        annotations = shift(annotations)
        if cost(annotations['corners']) < best_cost:
            best = annotations
            best_cost = cost(best['corners'])

    return best


if __name__ == "__main__":
    for image, labels in load_data(1):
        print(image.shape)
        print(labels)
