import json
from sys import argv
import cv2
import matplotlib.pyplot as plt

from utils.process import crop_board

piece_types = [
    'pawn_w',
    'pawn_b',
    'bishop_w',
    'bishop_b',
    'knight_w',
    'knight_b',
    'rook_w',
    'rook_b',
    'queen_w',
    'queen_b',
    'king_w',
    'king_b',
]


def label(file, dir):
    image = cv2.imread(file)
    size = min(image.shape[0], image.shape[1])
    x, y = (image.shape[0] - size) // 2, (image.shape[1] - size) // 2
    image = image[x:x + size, y:y + size]
    size = float(size)
    print(size)

    plt.imshow(image)
    plt.draw()
    plt.pause(0.01)

    def coords(text):
        while True:
            try:
                height, width = input(text).split(" ")
                return float(height) / size, 1 - float(width) / size
            except Exception as e:
                print(e)
                pass

    # corners: A8 H8 A1 H1
    # A1: WHITE LEFT
    # H1: WHITE RIGHT
    # A8: BLACK RIGHT
    # H8: BLACK LEFT

    a1 = coords('WHITE LEFT: ')
    h1 = coords('WHITE RIGHT: ')
    a8 = coords('BLACK RIGHT: ')
    h8 = coords('BLACK LEFT: ')
    corners = [a8, h8, a1, h1]
    print(corners)

    board = crop_board(image, corners, growthFactor=0.2, imageSize=size / 2)
    plt.imshow(board)
    plt.draw()
    plt.pause(0.01)

    config = {}
    while True:
        for piece in piece_types:
            squares = input(piece + ": ").upper().split(' ')
            for square in squares:
                if square == '':
                    continue

                config[square] = piece

        print('-------------')
        print(len(config))
        print(config)
        print('-------------')

        done = input('done? ')
        if done == 'y':
            break

    annotations = json.dumps({
        "config": config,
        "corners": corners
    })

    print(annotations)

    save = input('save? ')
    if save == 'y':
        number = input('number: ')
        cv2.imwrite(dir + number + '.jpg', image)
        open(dir + number + '.json', 'w').write(annotations)


if __name__ == '__main__':
    output = argv[1]
    files = argv[2:]

    for file in files:
        label(file, output)
