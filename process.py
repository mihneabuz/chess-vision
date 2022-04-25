import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2 
from sklearn.cluster import KMeans
from random import randint

R = (0, 0, 255)
B = (255, 0, 0)
G = (0, 255, 0)

attemps = 8
debug = False
corners = [1, 2, 65, 66]

def show_img(im):
    plt.imshow(im[:,:,::-1])
    plt.show()

def process_kmeans(im):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 150, 0.1, 30)
    points = corners[:, 0, :].astype(np.uint32)

    model = KMeans(n_clusters=64).fit(points)
    colors = [(randint(0, 256), randint(0, 256), randint(0, 256)) for _ in range(64)]
    for point, cluster in zip(points, model.predict(points)):
        print(point, cluster, colors[cluster])
        cv2.circle(im, point, 6, colors[cluster], -1)

    show_img(im)
    return gray

def closest_man(point, points):
    dist = 10000
    for x in points:
        d = abs(x[0] - point[0]) + abs(x[1] - point[1])
        if d < dist:
            dist = d

    return dist

def grid_cost(grid, points):
    cost = 0
    for point in grid:
        cost += closest_man(point, points)

    return cost

def gen_grid(start, diff):
    points = [start]
    rdiff = np.array([-diff[1], diff[0]])

    for i in range(-4, 5):
        if (i != 0):
            points.append(start + i * diff + i * rdiff)
            points.append(start + i * diff - i * rdiff)
        for j in range(-abs(i) + 1, abs(i)):
            points.append(start + i * diff + j * rdiff)
            points.append(start + i * rdiff + j * diff)

    return points

def process(im):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    mid = (h // 2, w // 2)
    features = cv2.goodFeaturesToTrack(gray, 110, 0.01, 30)
    points = sorted(features[:, 0, :].astype(np.int32), key=lambda p: (p[0] - mid[0]) ** 2 + (p[1] - mid[1]) ** 2)

    best_grid, best_cost = [], 1000000
    for i in range(attemps):
        start = points[i]

        for neigh in points[:attemps]:
            if neigh is not start:
                diff = neigh - start

                if abs(diff[0]) + abs(diff[1]) < 60:
                    continue

                if debug:
                    cv2.circle(im, neigh, 6, G, -1)
                    show_img(im)
                    cv2.circle(im, neigh, 6, R, -1)

                grid = gen_grid(start, diff)
                cost = grid_cost(grid, points)

                if (cost < best_cost):
                    best_cost = cost
                    best_grid = grid

                if debug:
                    copy = np.copy(im)
                    for point in grid:
                        cv2.circle(copy, point, 6, B, -1)
                    show_img(copy)
                    for point in grid:
                        cv2.circle(copy, point, 6, R, -1)

    print(best_cost)

    for point in points:
        cv2.circle(im, point, 6, R, -1)

    for point in best_grid:
        cv2.circle(im, point, 6, G, -1)

    for corner in corners:
        cv2.circle(im, best_grid[corner], 7, B, -1)

    show_img(im)

    return gray


def test():
    plt.rcParams["figure.figsize"] = (9,9)
    im = cv2.imread('boards/1010.jpg')
    # process(im)
    process(im)

if __name__ == '__main__':
    test()
