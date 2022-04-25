import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2 
from sklearn.cluster import KMeans
from random import randint

def process_kmeans(im):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 150, 0.1, 30)
    points = corners[:, 0, :].astype(np.uint32)

    model = KMeans(n_clusters=64).fit(points)
    colors = [(randint(0, 256), randint(0, 256), randint(0, 256)) for _ in range(64)]
    for point, cluster in zip(points, model.predict(points)):
        print(point, cluster, colors[cluster])
        cv2.circle(im, point, 6, colors[cluster], -1)

    plt.imshow(im[:,:,::-1])
    plt.show()

    return gray

def process(im):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 150, 0.1, 30)
    points = corners[:, 0, :].astype(np.uint32)

    for point in points:
        cv2.circle(im, point, 6, (0, 0, 255), -1)

    plt.imshow(im[:,:,::-1])
    plt.show()

    return gray


def test():
    plt.rcParams["figure.figsize"] = (10,10)
    im = cv2.imread('boards/1009.jpg')
    # process(im)
    process_kmeans(im)

if __name__ == '__main__':
    test()
