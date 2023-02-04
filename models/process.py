import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import trange

from utils import cells

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
boardSize = np.array([8, 8])
boardImgSize = np.ceil((boardSize * IMAGE_SIZE)/(1.0 + IMAGE_GROWTH_FACTOR * (boardSize - 1.0))).astype(int)
boardImgSize = tuple(boardImgSize)
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

def crop_board(image, corners):
    return unprojectCropFromRelativeCoords(image, corners, boardImgSize, IMAGE_GROWTH_FACTOR)

# TODO: study and cleanup these functions
marginsSize = np.array([1.0, 1.0]) * IMAGE_SIZE * IMAGE_GROWTH_FACTOR * 0.5
cellSize = (IMAGE_SIZE - 2.0 * marginsSize) / boardSize
cellRelSize = (1.0 - IMAGE_GROWTH_FACTOR) / boardSize
def getCellCenterRel(cellX, cellY):
    return np.zeros(2) + (IMAGE_GROWTH_FACTOR * 0.5) + np.array([0.5 + cellX, 0.5 + cellY]) * cellRelSize

def getCellBoundingBoxRel(cellX, cellY):
    cellCenter = getCellCenterRel(cellX, cellY)
    newExtents = np.dot(np.array([-0.5, 0.5]).reshape((2,1)), (cellRelSize + IMAGE_GROWTH_FACTOR).reshape((1,2)))
    return cellCenter + newExtents

def crop_pieces(image, pieces=None):
    images = []
    labels = []

    for cell in cells:
        cell_coords = cells[cell]
        cellBoundsRel = getCellBoundingBoxRel(cell_coords[0], cell_coords[1])
        cellBoundsAbs = np.round(np.multiply(cellBoundsRel.T, boardImgSize)).astype(int)
        
        piece_image = image[cellBoundsAbs[0,0]:cellBoundsAbs[0,1],cellBoundsAbs[1,0]:cellBoundsAbs[1,1]]

        images.append(piece_image)
        if pieces:
            labels.append(pieces.get(cell, 'empty'))

    image = None
    return images, labels

if __name__ == "__main__":
    from load_data import load_data
    
    for image, annotations in load_data(1):
        corners = annotations['corners']
        pieces = annotations['config']

        board_image = crop_board(image, corners)
        piece_images, cell_labels = crop_pieces(board_image, pieces)

        print(len(piece_images))
        print(len(cell_labels))

        plt.imshow(board_image)
        plt.show()

        for i in range(len(piece_images)):
            print(cell_labels[i])
            plt.imshow(piece_images[i])
            plt.show()
            break

