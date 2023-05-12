import numpy as np
import cv2


def dist(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def find_corners(mk, downscale=None, quality=0.3, infer=False):
    if downscale:
        mk = cv2.resize(mk, (downscale, downscale), interpolation=cv2.INTER_NEAREST)

    size = mk.shape[0]
    corners = cv2.goodFeaturesToTrack(mk, 4, quality, size / 4, useHarrisDetector=True)

    if corners is None or len(corners) > 4 or len(corners) < 3:
        return np.array([])

    corners = list(corners[:, 0, :].astype(np.float64) / size)
    if len(corners) == 3:
        if infer:
            final = infer_last_corner(corners, mk)
            corners = [corners[0], corners[1], corners[2], final]
        else:
            return np.array([])

    return sort_corners(corners)


def infer_last_corner(corners, mk):
    size = mk.shape[0]

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
    return find_nearest(guess, mk) / size


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
