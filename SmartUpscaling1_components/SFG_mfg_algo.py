#imports
from PIL import Image

import numpy as np
import cv2
import math
from os import path

#variables
weight_old =  1.2917347694671983
weight_new =  -0.29173476946719834

weight_old, weight_new = weight_new, weight_old

to_generate = 41 #numéro de la derniere image à generer
mode = 2 #mode de multi-génération (2 --> 2 images, 2x, SFG.MFG | 1 --> 1 image, 1x, SFG.FG)
i=0


def generate_interframe(prev_image, next_image):
    #créer une image vide
    current_image = np.zeros((prev_image.shape[0], prev_image.shape[1], 3), np.uint8)

    #faire la moyenne de l'image précédente et de l'image suivante
    current_image = cv2.addWeighted(prev_image, weight_old, next_image, weight_new, 0)

    return current_image

while (i <= to_generate):
    if mode == 2:
        #charger l'image précédente
        prev_image = cv2.imread("./data/lowres_frames_2x/frame"+str(i)+".jpg")
        #charger l'image suivante
        next_image = cv2.imread("./data/lowres_frames_2x/frame"+str(i+3)+".jpg")

        #créer l'image intermédiaire droite
        current_image_right = generate_interframe(prev_image, next_image)

        #créer l'image intermédiaire gauche
        current_image_left = generate_interframe(prev_image, current_image_right)

        #enregistrer l'image intermédiaire droite
        cv2.imwrite("./data/lowres_generated_frames_2x/frame"+str(i+2)+".jpg", current_image_right) #enregistrer l'image (dans data/lowres_generated_frames)

        #enregistrer l'image intermédiaire droite
        cv2.imwrite("./data/lowres_generated_frames_2x/frame"+str(i+1)+".jpg", current_image_left) #enregistrer l'image (dans data/lowres_generated_frames)

    elif mode == 1:
        #charger l'image précédente
        prev_image = cv2.imread("./data/lowres_frames_1x/frame"+str(i)+".jpg")
        #charger l'image suivante
        next_image = cv2.imread("./data/lowres_frames_1x/frame"+str(i+2)+".jpg")

        #créer l'image intermédiaire
        current_image = generate_interframe(prev_image, next_image)

        #enregistrer l'image intermédiaire
        cv2.imwrite("./data/lowres_generated_frames_1x/frame"+str(i+1)+".jpg", current_image) #enregistrer l'image (dans data/lowres_generated_frames)

    #mettre à jour les variables
    i += mode+1