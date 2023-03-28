from random import random
from sys import argv

import numpy as np
import matplotlib.pyplot as plt
import cv2

from board_detection import Service as BoardDetection
from piece_classification import Service as PieceClassification
from utils.process import crop_board, crop_pieces
from utils.utils import deserialize_array, serialize_array, deserialize_values, image_from_bytes
from utils.load_data import sort_corners

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

def unstack_corners(corners):
    return [x for corner in corners for x in corner]

def print_board(pieces):
    BOLD = '\033[1m'
    END = '\033[0m'
    GRAY = '\033[90m'
    RED = '\033[91m'

    pieces_map = [' ', 'p', 'P', 'b', 'B', 'n', 'N', 'r', 'R', 'q', 'Q', 'k', 'K']
    pieces_map = [RED + piece + END if i % 2 == 0 else BOLD + piece + END for i, piece in enumerate(pieces_map)]

    for i in range(8):
        print(GRAY + '----------------------------------' + END)
        for j in range(8):
            print(GRAY + '|' + END, end=' ')
            print(pieces_map[pieces[i * 8 + j]], end=' ')
        print(GRAY + '|' + END)
    print(GRAY + '----------------------------------' + END)


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
    result1 = board_detection.process([input1, input1])

    corners = stack_corners(deserialize_array(result1[0]))
    print(corners)

    #  DELETE:
    import json
    corners = json.load(open(file.replace('jpg', 'json')))['corners']
    """ result1[0] = [max([x, 0]) for x in result1[0]] """
    corners = sort_corners(corners)
    plt.imshow(add_corners(image, corners))
    plt.show()

    cropped = crop_board(image, corners)
    pieces, _ = crop_pieces(cropped)
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(4, 4))
    for i, axi in enumerate(ax.flat):
        axi.imshow(pieces[int(random() * 64)])
    plt.tight_layout()
    plt.show()

    input2 = (image_bytes, serialize_array(np.array(unstack_corners(corners)).astype(np.float32)))
    result2 = piece_classification.process([input2, input2])

    found = deserialize_values(result2[0], 64, np.uint8)
    print_board(found)

    plt.imshow(cropped)
    plt.show()
