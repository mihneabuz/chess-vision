import numpy as np
import matplotlib.pyplot as plt
from cv2 import imread
import cv2

def process(im):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 128, 0.1, 30)
    for corner in corners:
        point = corner[0].astype(np.uint32)
        print(point)
        cv2.circle(im, point, 6, (0, 0, 255), -1)

    plt.imshow(im[:,:,::-1])
    plt.show()

    return gray

def test():
    print('testing crop...')
    plt.rcParams["figure.figsize"] = (10,10)
    im = imread('boards/1003.jpeg')
    plt.imshow(im[:,:,::-1])
    plt.show()
    im = process(im)

if __name__ == '__main__':
    test()
