from random import random
from sys import argv

import numpy as np
import matplotlib.pyplot as plt
import cv2

from board_detection import Service as BoardDetection
from piece_classification import Service as PieceClassification
from utils.process import crop_board, crop_pieces
from utils.utils import deserialize_array, image_from_bytes

size = 160

def add_corners(image, corners):
    image = np.copy(image)
    (height, width, _) = image.shape

    cv2.circle(image, [int(corners[0][1] * width), int(corners[0][0] * height)], 20, (255, 0, 0), 4)
    cv2.circle(image, [int(corners[1][1] * width), int(corners[1][0] * height)], 20, (0, 255, 0), 4)
    cv2.circle(image, [int(corners[2][1] * width), int(corners[2][0] * height)], 20, (0, 0, 255), 4)
    cv2.circle(image, [int(corners[3][1] * width), int(corners[3][0] * height)], 20, (0, 255, 255), 4)

    return image

def stack_corners(corners):
    return [
        [corners[0], corners[1]],
        [corners[2], corners[3]],
        [corners[4], corners[5]],
        [corners[6], corners[7]],
    ]


if __name__ == '__main__':
    file = argv[1]

    board_detection = BoardDetection()
    piece_classification = PieceClassification()

    with open('./board_detection_weights', 'rb') as w:
        board_detection.load_model(w.read())

    with open('./piece_classification_weights', 'rb') as w:
        piece_classification.load_model(w.read())

    image_bytes = None
    with open(file, 'rb') as image:
        image_bytes = image.read()

    image = image_from_bytes(image_bytes)
    print(image.shape)

    input1 = (image_bytes, bytes())
    result1 = board_detection.process([input1])

    corners = stack_corners(deserialize_array(result1[0]))
    print(corners)

    #  DELETE:
    import json
    corners = json.load(open('boards/data/17.json'))['corners']

    plt.imshow(add_corners(image, corners))
    plt.show()

    cropped = crop_board(image, corners)
    plt.imshow(cropped)
    plt.show()

    pieces, _ = crop_pieces(cropped)
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(4, 4))
    for i, axi in enumerate(ax.flat):
        axi.imshow(pieces[int(random() * 64)])
    plt.tight_layout()
    plt.show()

    input2 = ()
