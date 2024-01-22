import cv2
import numpy as np
import math


'''Metodo para buscar el color predominate (hsv)'''
def find_dominant_color_hsv(img):      
    # Reshape the image to be a list of pixels
    pixels = img.reshape(-1, 3)
    # Convert to float32 for k-means clustering
    pixels = np.float32(pixels)
    # Perform k-means clustering
    k = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convert the center color from BGR to RGB format
    bgr_dominant_color = centers[0].astype(np.uint8)
    rgb_dominant_color = centers[0].astype(np.uint8)[::-1]
    hsv_dominant_color = cv2.cvtColor(np.array([[rgb_dominant_color]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0][0]
    # Retur hsv dominant color
    return tuple(hsv_dominant_color)

'''Metodo para buscar el color predominate (lab)'''
def find_dominant_color_lab(img):      
    # Reshape the image to be a list of pixels
    pixels = img.reshape(-1, 3)
    # Convert to float32 for k-means clustering
    pixels = np.float32(pixels)
    # Perform k-means clustering
    k = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convert the center color from BGR to RGB format
    bgr_dominant_color = centers[0].astype(np.uint8)
    rgb_dominant_color = centers[0].astype(np.uint8)[::-1]
    lab_dominant_color = cv2.cvtColor(np.array([[rgb_dominant_color]], dtype=np.uint8), cv2.COLOR_RGB2Lab)[0][0]
    # Retur lab dominant color
    return tuple(lab_dominant_color)

'''Metodo para sacar la distancia euclidea entre dos puntos'''
def euclidean_distance(center1, center2):
    return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)