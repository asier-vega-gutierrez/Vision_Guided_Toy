import cv2
import numpy as np

'''Clase para recuperar el tablero y solo el tablero de la imagen'''
class FindBoard:

    def __init__(self) -> None:
        self.img =  None
        self.img_draw = None
        self.img_board = None
        self.img_no_board = None
        self.img_cropped = None
        self.img_blurred = None
        self.img_thresholded = None
        self.contours = None
        self.img_contours = None
    
    '''Metodo principal para buscar y recuperar el tablero'''
    def find_board(self, img):
        self.img = img.copy()
        self.img_draw = img.copy()
        #Preprocesar la imagen
        self._preproces()
        #Buscar el controno mas grande (es el del tablero)
        max_contour = max(self.contours, key=cv2.contourArea)
        #Pintar el contorno en la iamagen
        self.img_draw = cv2.drawContours(self.img_draw, [max_contour], -1, (0, 255, 0), 1)
        #Pertiendo de una imagen de zeros
        mask = np.zeros_like(self.img)
        #Se pinta el contorno relleno sobre la mascara
        cv2.drawContours(mask, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        #Se utiliza la mascara para hacer recuperar la parte blanca pero de la imagen oprginal y el resto dejarlo en negro
        self.img_board = cv2.bitwise_and(img, mask)
        #Tambien hacemos la operacion contraria para obtener la imagen sin tablero
        inverted_mask = cv2.bitwise_not(mask)
        self.img_no_board = cv2.bitwise_and(img, inverted_mask)
        #Devolvemos el tablero
        return self.img_board, self.img_no_board, self._find_centrer(max_contour), max_contour

    '''Metodo de preprocesado'''
    def _preproces(self):
        #Pasar la imagen a escala de grises
        img_gray = cv2.cvtColor(self.img_draw, cv2.COLOR_BGR2GRAY)
        #Dar blurr a la imagen
        self.img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        #Threshold adaptativo
        block_size = 5  # Size of the pixel neighborhood used for adaptive thresholding
        c = 3  # Constant subtracted from the mean
        self.img_thresholded = cv2.adaptiveThreshold(self.img_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, c)
        #Deteccion de bordes
        self.contours, _ = cv2.findContours(self.img_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.img_contours = cv2.drawContours(self.img_draw, self.contours, -1, (255,255,0), 2)

    '''Metodo para encontra el centro de un poligono'''
    def _find_centrer(self, contour):
        #Se crea el circulo mas peqeuño que contenga al poligono, y se coje el centro del de ese ciculo
        (x, y), _ = cv2.minEnclosingCircle(contour)
        return (int(x), int(y)) 