from numba import cuda
import cupy as cp
import numpy as np
from PIL import Image, ImageTk
from tkinter import *
import random
import math
import time
from tkinter import *
from PIL import Image, ImageTk





@cuda.jit
def computeImage(A, B):
    row, col = cuda.grid(2)
    if row < A.shape[0] and col < A.shape[1]:
        z = A[row, col, 0] + A[row, col, 1] * 1j
        c = z
        iterations = 0
        while iterations < 256:
            z = z * z + c
            if abs(z) > 2:
                B[row, col, 0] = iterations
                B[row, col, 1] = iterations  
                B[row, col, 2] = iterations
                break
            elif iterations == 255:
                B[row, col, 0] = 255
                B[row, col, 1] = 255
                B[row, col, 2] = 255
            iterations +=1
        



def makeImage(data):
    im = Image.new("RGB", imageSize)
    pixel_data = data.reshape(-1, 3).astype(int).tolist()
    x = list(map(tuple, pixel_data))
    im.putdata(x)
    showcanvas = ImageTk.PhotoImage(im)
    canvas.create_image(0, 0, anchor=NW, image=showcanvas)
    canvas.image = showcanvas  # Keep a reference to avoid garbage collection
    canvas.pack()

def calcPixleCoordiantes():
    return np.array([[( xMin + (xMax - xMin)/imageSize[0] * x, yMin + (yMax - yMin)/imageSize[0] * y) for x in range(imageSize[0])] for y in range(imageSize[1])], dtype=np.float64)

def main():
    start = time.time()
    pixleCoords = calcPixleCoordiantes()
    imageData = cp.zeros((imageSize[0], imageSize[1], 3), dtype=np.uint8)
    print(imageData)
    threadsperblock = (32, 32)  
    blockspergrid_x = int(np.ceil(pixleCoords.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(pixleCoords.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)  
    print(f"The kernel will be executed up to element {threadsperblock[0]*blockspergrid_x}")
    computeImage[blockspergrid, threadsperblock](pixleCoords, imageData)
   

    

    makeImage(imageData)
    end = time.time()
    print(f"Math Done Time = {end-start}")

if __name__ == '__main__':
    xMax = 1
    xMin = -2
    yMax = 1
    yMin = -1
    #cuda.detect()
    imageSize = (1024, 1024)
    
    win = Tk()
    win.geometry("1200x1200")
    canvas= Canvas(win, width= 5000, height= 5000)
    canvas.pack()

    main()
    win.mainloop()

    