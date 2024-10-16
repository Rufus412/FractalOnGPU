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
import unittest





@cuda.jit
def computeImage(A, B, C):
    row, col = cuda.grid(2)
    if row < A.shape[0] and col < B.shape[1]:
        z = A[row, col] + B[row, col] * 1j
        c = z
        iterations = 0
        while iterations < 256:
            z = z * z + c
            if abs(z) > 2:
                combined = (iterations + 15 << 16) | (iterations << 8) | 2 * iterations + 10 
                C[row, col] = combined
                break
            iterations +=1
        if iterations == 255:
            C[row, col] = 0
        



def makeImage(data):
    im = Image.new("RGB", imageSize)
    im.putdata(data.flatten().tolist())
    showcanvas = ImageTk.PhotoImage(im)
    canvas.create_image(0, 0, anchor=NW, image=showcanvas)
    canvas.image = showcanvas  # Keep a reference to avoid garbage collection
    canvas.pack()
    im.save('pillow_imagedraw.png')

def calcPixleCoordiantes():
    #return cp.array([[( xMin + (xMax - xMin)/imageSize[0] * x, yMin + (yMax - yMin)/imageSize[0] * y) for x in range(imageSize[0])] for y in range(imageSize[1])], dtype=np.float64)
    x_space = np.linspace(xMin, xMax,imageSize[0], dtype=np.float64)
    y_space = np.linspace(yMin, yMax,imageSize[1], dtype=np.float64)
    return np.meshgrid(x_space, y_space)

def main():
    start = time.time()
    start1 = time.time()
    pixleCoords_x, pixleCoords_y = calcPixleCoordiantes()
    end1 = time.time()
    print(f"generate coords took {end1-start1} seconds")
    #imageData = cp.zeros((imageSize[0], imageSize[1], 3), dtype=np.uint8)
    imageData = cp.zeros((imageSize[0], imageSize[1]), dtype=np.int32)
    #print(imageData)
    threadsperblock = (32, 32)  
    blockspergrid_x = int(np.ceil(pixleCoords_x.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(pixleCoords_y.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)  
    #print(f"The kernel will be executed up to element {threadsperblock[0]*blockspergrid_x}")
    
    start2 = time.time()
    computeImage[blockspergrid, threadsperblock](pixleCoords_x, pixleCoords_y, imageData)
   
    end2 = time.time()
    print(f"Compute fractal took {end2-start2} seconds")
    
    start3 = time.time()
    makeImage(imageData)
    end3 = time.time()
    print(f"Make Image took {end3-start3} seconds")
    end = time.time()
    print(f"Time = {end-start}")

if __name__ == '__main__':
    xMax = 1
    xMin = -2
    yMax = 1
    yMin = -1
    #cuda.detect()
    imageSize = (4096, 4096)
    
    win = Tk()
    win.geometry("1200x1200")
    canvas= Canvas(win, width= 1024, height= 1024)
    canvas.pack()

    main()
    win.mainloop()

    