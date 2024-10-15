from numba import cuda
import cupy as cp
import numpy as np





@cuda.jit
def computeImage(A, B):
    row, col = cuda.grid(2)
    if row < A.shape[0] and col < A.shape[0]:
        z = A[row][col][0] + A[row][col][1] * 1j



#def segmentImage():

def calcPixleCoordiantes():
    return np.array([[( xMin + (xMax - xMin)/imageSize[0] * x, yMin + (yMax - yMin)/imageSize[0] * y) for x in range(imageSize[0])] for y in range(imageSize[1])], dtype=np.float64)

def main():
    pixleCoords = calcPixleCoordiantes()
    imageData = cp.zeros((imageSize[0], imageSize[1]), dtype=np.int32)
    print(imageData)
    threadsperblock = (128, 128)  # each block will contain 16x16 threads, typically 128 - 512 threads/block
    blockspergrid_x = int(np.ceil(pixleCoords.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(pixleCoords.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)  # we calculate the gridsize (number of blocks) from array
    print(blockspergrid)
    print(f"The kernel will be executed up to element {threadsperblock[0]*blockspergrid_x}")

if __name__ == '__main__':
    xMax = 2
    xMin = -2
    yMax = 1
    yMin = -1
    #cuda.detect()
    imageSize = (1024, 1024)


    
    
    main()

    