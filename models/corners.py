import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.load_data import load_data
from utils.utils import dataset


def load_datasets(limit=-1):
    size = 480
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


def find_corners(mk):
    size = mk.shape[0]
    edges = cv2.Canny(mk, 50, 150)
    corners = cv2.goodFeaturesToTrack(edges, 4, 0.5, size / 4).astype(np.int32)

    if len(corners) == 3:
        # TODO: infer 4th corner if missing
        pass

    return corners


def debug_corners(im, mk):
    size = im.shape[0]

    im[mk == 0] = (im[mk == 0].astype(np.float32) * 0.3).astype(np.uint8)

    edges = cv2.Canny(mk, 50, 150)
    im[edges > 0.5] = np.array([255, 255, 0])

    corners = cv2.goodFeaturesToTrack(edges, 4, 0.5, size / 4).astype(np.int32)
    for corn in corners:
        c = corn[0]
        cv2.circle(im, (c[0], c[1]), 8, (255, 255, 0), 1)

    plt.imshow(im)
    plt.show()


if __name__ == '__main__':
    for im, (mk, c) in load_datasets(limit=2):
        print(im.shape, im.dtype)
        print(mk.shape, mk.dtype)

        cv2.circle(im, (c[0][0], c[0][1]), 3, (255, 0, 0), 3)
        cv2.circle(im, (c[1][0], c[1][1]), 3, (0, 255, 0), 3)
        cv2.circle(im, (c[2][0], c[2][1]), 3, (0, 0, 255), 3)
        cv2.circle(im, (c[3][0], c[3][1]), 3, (0, 255, 255), 3)

        debug_corners(im, mk)
