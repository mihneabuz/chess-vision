import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt

IMAGE_GROWTH_FACTOR = 0.1
IMAGE_SIZE = 128

# TODO: study and cleanup this function
def getSegmentsIntersection(a1, a2, b1, b2):
    a = a2 - a1
    b = b2 - b1
    o = a1 - b1
    aLenSq = np.dot(a, a)
    aDotB = np.dot(a, b)
    denom = (aLenSq * np.dot(b, b)) - aDotB**2.0

    if denom == 0.0:
        # The segment are parallel, no unique intersection exists
        return None

    u = ((aLenSq * np.dot(b, o)) - (aDotB * np.dot(a, o))) / denom

    if u < 0.0 or u > 1.0:
        # The potential intersection point is not situated on the segment, aborting
        return None

    t = np.dot(a, u * b - o) / aLenSq
    aPoint = a1 + t * a
    bPoint = b1 + u * b

    return aPoint if (np.linalg.norm(np.round(aPoint - bPoint), 5) == 0.0) else None

# TODO: study and cleanup this function
def unprojectCropFromRelativeCoords(img_, cx, outputShape, growthFactor = 0.0):
    boardCornersRel = np.array(cx)
    boardCornersRel = np.column_stack((boardCornersRel[:,0], 1.0 - boardCornersRel[:,1]))

    O = getSegmentsIntersection(boardCornersRel[0], boardCornersRel[3], boardCornersRel[1], boardCornersRel[2])
    for i in range(4):
        boardCornersRel[i] = growthFactor * (boardCornersRel[i] - O) + boardCornersRel[i]

    cornersAbs = np.round(boardCornersRel * (np.array(img_.shape)[0:2] - np.array([1.0, 1.0])))

    outCoords = np.array([
        [outputShape[0]-1, 0.0],
        [outputShape[0]-1, outputShape[1]-1],
        [0.0, 0.0],
        [0.0, outputShape[1]-1],
    ])

    inCoords = np.float32(cornersAbs)
    outCoords = np.float32(outCoords)
    M = cv2.getPerspectiveTransform(inCoords, outCoords)

    return cv2.warpPerspective(img_, M, outputShape, flags=cv2.INTER_LINEAR)

def crop_image(image, corners):
    boardSize = np.array([8, 8])
    boardImgSize = np.ceil((boardSize * IMAGE_SIZE)/(1.0 + IMAGE_GROWTH_FACTOR * (boardSize - 1.0))).astype(int)
    boardImgSize = tuple(boardImgSize)
    
    return unprojectCropFromRelativeCoords(image, corners, boardImgSize, IMAGE_GROWTH_FACTOR)

if __name__ == "__main__":
    from load_data import load_data, get_corners 
    images, labels = load_data(1)
    corners = get_corners(labels)
    image = images[0]
    corns = corners[0]

    board_image = crop_image(image, corns)

    plt.imshow(board_image, cmap='gray')
    plt.show()

