from cv2 import cv2
import json
from os import listdir
from random import sample

DATA = 'boards'

def load_data(max=-1):
    files = [file[:-4] for file in listdir(DATA) if file.endswith('.jpg')]

    if (max > 0):
        files = sample(files, max)

    images = [cv2.imread(DATA + '/' + file + '.jpg') for file in files]
    labels = [json.load(open(DATA + '/' + file + '.json')) for file in files]

    return images, labels

def get_corners(labels):
    return [label['corners'] for label in labels]

def get_pieces(labels):
    return [label['config'] for label in labels]

if __name__ == "__main__":
    images, labels = load_data(1)
    corners = get_corners(labels)
    pieces = get_pieces(labels)
    print(images[0].shape)
    print(corners[0])
    print(pieces[0])
