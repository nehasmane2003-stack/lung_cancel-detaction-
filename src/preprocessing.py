import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    return img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)