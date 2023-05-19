from random import random
from sys import argv

import numpy as np
import matplotlib.pyplot as plt
import cv2
import gc as gc

from board_detection import Service as BoardDetection
from piece_classification import Service as PieceClassification
from board_segmentation import Service as BoardSegmentation
from utils.process import crop_board, crop_pieces, translate_pieces
from utils.utils import deserialize_array, deserialize_values, image_from_bytes, image_to_bytes, classes_dict


def add_corners(image, corners):
    image = np.copy(image)
    (height, width, _) = image.shape

    cv2.circle(image, [int(corners[0][0] * width), int(corners[0][1] * height)], 20, (255, 0, 0), 4)
    cv2.circle(image, [int(corners[1][0] * width), int(corners[1][1] * height)], 20, (0, 255, 0), 4)
    cv2.circle(image, [int(corners[2][0] * width), int(corners[2][1] * height)], 20, (0, 0, 255), 4)
    cv2.circle(image, [int(corners[3][0] * width), int(corners[3][1] * height)], 20, (0, 255, 255), 4)

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


def visualize_single(models):
    [board_detection, board_segmentation, piece_classification] = models
    file = argv[2]

    image_bytes = None
    with open(file, 'rb') as image:
        image_bytes = image.read()

    image = image_from_bytes(image_bytes)
    print(image.shape)

    input1 = (image_bytes, bytes())
    det_corners = board_detection.process([input1, input1])
    plt.imshow(add_corners(image, stack_corners(deserialize_array(det_corners[0]))))
    plt.show()

    seg_corners = board_segmentation.process([input1, input1])
    if len(seg_corners[0]) == 0:
        return

    plt.imshow(add_corners(image, stack_corners(deserialize_array(seg_corners[0]))))
    plt.show()

    corners = stack_corners(deserialize_array(seg_corners[0]))
    print(corners)

    cropped = crop_board(image, corners, flag=True)
    pieces, _ = crop_pieces(cropped)
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(4, 4))
    for i, axi in enumerate(ax.flat):
        axi.imshow(pieces[int(random() * 64)])
    plt.tight_layout()
    plt.show()

    input3 = (image_bytes, seg_corners[0])
    result3 = piece_classification.process([input3, input3])

    found = deserialize_values(result3[0], 64, np.uint8)
    print(found)
    print_board(found)

    plt.imshow(cropped)
    plt.show()


def evaluate(models):
    [_, board_segmentation, piece_classification] = models
    count = int(argv[2])

    from utils.load_data import load_data

    images = []
    for image, annotations in load_data(max=count):
        pieces = [classes_dict[label] for label in translate_pieces(annotations["config"])]
        images.append((image_to_bytes(image), pieces))

    gc.collect()

    input1 = [(image[0], bytes()) for image in images]
    result1 = board_segmentation.process(input1)

    errors = len([0 for res in result1 if len(res) == 0])
    images = [images[i] for i in range(len(images)) if len(result1[i]) > 0]
    print('Unable to find corners: ', errors)

    input2 = [(images[i][0], result1[i]) for i in range(len(images)) if len(result1[i]) > 0]
    result2 = piece_classification.process(input2)

    errors = len([0 for res in result2 if len(res) == 0])
    images = [images[i] for i in range(len(images)) if len(result2[i]) > 0]
    print('Unable to crop board: ', errors)

    found = [deserialize_values(data, 64, np.uint8) for data in result2 if len(data) > 0]
    for i in range(len(images)):
        real = images[i][1]
        pred = list(reversed(found[i].tolist()))
        print(pred)
        print(real)

        count = len(list(filter(lambda a: a[0] == a[1], zip(real, pred))))

        counts = [0 for _ in range(13)]
        for x in pred:
            counts[x] += 1

        for x in real:
            counts[x] -= 1

        print('accuracy: ', count / 64)
        print('piece count diff: ', counts)
        print('--------------------------------------------------------------')


def segmentation():
    file = argv[2]

    image = cv2.imread(file)
    print(image.shape)
    size = image.shape[0]

    plt.imshow(image)
    plt.show()

    import board_segmentation as seg
    model = seg.create_model(load_dict=True)
    model.eval()

    input = seg.jit_transform(seg.tensor_transform(image))
    mask = model(input[None, :, :, :])[0].detach().numpy()[0, 0, :, :]
    plt.imshow(mask)
    plt.show()

    mask = cv2.resize(mask, (size, size))
    image = image / 255
    image[mask < 0.9] *= np.array([0.2, 0.1, 0.0])
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    mode = argv[1]

    board_detection = BoardDetection()
    board_segmentation = BoardSegmentation(maskSize=100, quality=0.3)
    piece_classification = PieceClassification()

    with open('./board_detection_weights', 'rb') as w:
        board_detection.load_model(w.read())

    with open('./board_segmentation_weights', 'rb') as w:
        board_segmentation.load_model(w.read())

    with open('./piece_classification_weights', 'rb') as w:
        piece_classification.load_model(w.read())

    models = [board_detection, board_segmentation, piece_classification]

    if mode == "single":
        visualize_single(models)

    if mode == "eval":
        evaluate(models)

    if mode == "segmentation":
        segmentation()
