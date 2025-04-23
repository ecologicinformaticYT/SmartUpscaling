"""
SmartUpscaling2
--> SmartUpscaling Optimised Optical Flow 1
(component for OptimisedZonesCalculator1, SmartFrameGeneration2, SmartFramePrediction1, SmartColorsPrediction1, SmartFrameReconstruction1)

2 modes :
- normal --> kernel_normal.c
- AI boosted --> kernel_ai.c
"""
import numpy as np
from PIL import Image

def computeOpticalFlow(image, mode, kernel_args=[]):
    pass

def plotObjects(image, mode, kernel_args=[], dbg_color=(255,0,0,255)):
    #objects = computeOpticalFlow(image, mode, kernel_args)
    objects = [[(0,0), (10,10)], [(0,10), (5,-5)]]
    image_copy = Image.open(image).convert("RGBA").copy() #copy the RGBA image
    pixels=[]
    if objects is not None:
        for object_ in objects:
            #print(object_)
            for a, b in object_:
                vector = ( #get the vector between A and B
                    b[0] - a[0],
                    b[1] - a[1],
                )
                #print(vector)
                repr_ = { #parametric representation of the line
                    "x":{"start":a[0], "coef":vector[0]},
                    "y":{"start":a[1], "coef":vector[1]},
                }
                #print(repr_)
                t = 0
                while not (repr_["x"]["start"]+repr_["x"]["coef"]*t, repr_["y"]["start"]+repr_["y"]["coef"]*t) == b:
                    if  (repr_["x"]["start"]+repr_["x"]["coef"]*t, repr_["y"]["start"]+repr_["y"]["coef"]*t) == b:
                        break
                    t += 1
                    print(repr_["x"]["start"]+repr_["x"]["coef"]*t, repr_["y"]["start"]+repr_["y"]["coef"]*t)
                print(t)

                #get every pixel between a and b
                for i in range(0, t+1):
                    pixels.append((repr_["x"]["start"]+repr_["x"]["coef"]*i, repr_["y"]["start"]+repr_["y"]["coef"]*i))
                #print(pixels)

                #draw the line
                for pixel in pixels:
                    image_copy.putpixel(pixel, dbg_color)

    #image_copy.show()


if __name__ == "__main__":
    params = {"k":np.int8(5), "treshold":np.float32(5/100), "k1_range":np.array([1, 5], dtype=np.int8), "k2_range":np.array([1, 5], dtype=np.int8)}
    #print(params)
    plotObjects("tests/image.jpg", "normal", params)