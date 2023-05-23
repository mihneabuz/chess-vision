import json
from sys import argv
import cv2
import matplotlib.pyplot as plt

from utils.process import cells, crop_board, crop_pieces
from piece_classification import classes_dict
from test import print_board


def check(file):
    image = cv2.imread(file)
    annotations = json.load(open(file[:-3] + 'json'))

    corners = annotations['corners']
    pieces = annotations['config']

    board_image = crop_board(image, corners)
    piece_images, cell_labels = crop_pieces(board_image, pieces=pieces)

    print('### FILE: ' + file + ' ###')
    table = [[0 for _ in range(8)] for _ in range(8)]
    for cell, piece in pieces.items():
        [i, j] = cells[cell]
        table[i][j] = classes_dict[piece]
    print_board([piece for line in table for piece in line])

    plt.imshow(board_image)
    plt.show()


if __name__ == '__main__':
    files = argv[1:]

    for file in files:
        check(file)
