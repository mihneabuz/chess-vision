import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt

from utils.load_data import load_data
from utils.utils import dataset


def dist(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_corners(mk):
    size = mk.shape[0]
    edges = cv2.Canny(mk, 50, 150)
    corners = cv2.goodFeaturesToTrack(edges, 4, 0.4, size / 4).astype(np.int32)

    if corners is None or len(corners) > 4:
        return np.array([])

    if len(corners) < 3:
        return np.array(corners)

    corners = list(corners[:, 0, :].astype(np.float64) / size)
    if len(corners) == 3:
        d1 = dist(corners[0], corners[1])
        d2 = dist(corners[1], corners[2])
        d3 = dist(corners[0], corners[2])

        if d1 > d2 and d1 > d3:
            near = [corners[0], corners[1]]
            opposite = corners[2]

        if d2 > d1 and d2 > d3:
            near = [corners[1], corners[2]]
            opposite = corners[0]

        if d3 > d1 and d3 > d2:
            near = [corners[0], corners[2]]
            opposite = corners[1]

        center = (near[0] + near[1]) / 2
        guess = 2 * center - opposite

        guess = (guess * size).astype(np.int32)
        final = find_nearest(guess, edges)
        corners = [corners[0], corners[1], corners[2], final / size]

    return sort_corners(corners)


def find_nearest(point, mk):
    size = mk.shape[0]

    def is_edge(x, y):
        return x >= 0 and x < size and y >= 0 and y < size and mk[x, y] > 0.5

    x, y = point[1], point[0]
    if is_edge(x, y):
        return point

    i = 1
    while i < size / 40:
        for j in range(0, i):
            if is_edge(x + j, y + i - j):
                return np.array([y + i - j, x + j])

            if is_edge(x + j, y - i + j):
                return np.array([y - i + j, x + j])

            if is_edge(x - j, y + i - j):
                return np.array([y + i - j, x - j])

            if is_edge(x - j, y - i + j):
                return np.array([y - i + j, x - j])

        i += 1

    return point


def sort_corners(corners):
    def key(c):
        return c[0] + c[1]

    corners.sort(key=lambda x: x[0] + x[1])
    bot_right = corners[3]
    top_left = corners[0]

    corners = corners[1:3]
    corners.sort(key=lambda x: x[0] - x[1])
    bot_left = corners[0]
    top_right = corners[1]

    return np.array([top_right, bot_right, top_left, bot_left])


def debug_corners(im, mk):
    size = im.shape[0]

    im[mk == 0] = (im[mk == 0].astype(np.float32) * 0.3).astype(np.uint8)

    edges = cv2.Canny(mk, 50, 150)
    im[edges > 0.5] = np.array([255, 0, 0])

    corners = find_corners(mk)
    if len(corners) == 4:
        cv2.circle(im, (int(corners[0][0] * size), int(corners[0][1] * size)), 6, (255, 0, 0), 1)
        cv2.circle(im, (int(corners[1][0] * size), int(corners[1][1] * size)), 6, (0, 255, 0), 1)
        cv2.circle(im, (int(corners[2][0] * size), int(corners[2][1] * size)), 6, (0, 0, 255), 1)
        cv2.circle(im, (int(corners[3][0] * size), int(corners[3][1] * size)), 6, (0, 255, 255), 1)

    return im


if __name__ == '__main__':
    def load_datasets(limit=-1):
        size = 280
        images = []
        masks = []
        corners = []
        for image, annotations in load_data(max=limit):
            images.append(cv2.resize(image, (size, size)))

            mask = np.zeros((size, size), dtype=np.float32)
            c = list(map(lambda x: [x[0], 1 - x[1]], annotations['corners']))
            c = np.array([c[0], c[1], c[3], c[2]])
            cv2.fillPoly(mask, pts=[(c * size).astype(np.int32)], color=1)
            masks.append((mask * 255).astype(np.uint8))

            corners.append((np.array(c) * size).astype(np.int32))

        print(f'loaded {len(images)} images')
        print(f'image size: {images[0].shape}')

        return dataset(images, list(zip(masks, corners)))

    for im, (mk, c) in load_datasets(limit=100):
        print(im.shape, im.dtype)
        print(mk.shape, mk.dtype)

        im = debug_corners(im, mk)

        cv2.circle(im, (c[0][0], c[0][1]), 2, (255, 0, 0), 3)
        cv2.circle(im, (c[1][0], c[1][1]), 2, (0, 255, 0), 3)
        cv2.circle(im, (c[2][0], c[2][1]), 2, (0, 0, 255), 3)
        cv2.circle(im, (c[3][0], c[3][1]), 2, (0, 255, 255), 3)

        plt.imshow(im)
        plt.show()
