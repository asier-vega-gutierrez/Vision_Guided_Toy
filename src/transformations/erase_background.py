import cv2
import numpy as np

'''Clase para sliminar el fondo segun color predominante (hsv)'''
class EraseBackground():

    def __init__(self) -> None:
        self.img = None
        self.img_draw = None
        
    '''Metodo para borrar el fondo segundo color predomientes'''
    def erase_background(self, img, color_type: int, tolerance: int):
        self.img = img
        
        #Definir el color que se quiere eliminar
        if color_type == 0:
            img_colored = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            color_to_remove = self._find_dominant_color_hsv()
        elif color_type == 1:
            img_colored = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
            color_to_remove = self._find_dominant_color_lab()

        #Se calculan los rangos alto y bajo segun el parametro de entrada
        lower_bound = (color_to_remove[0] - tolerance, color_to_remove[1] - tolerance, color_to_remove[2] - tolerance)
        upper_bound = (color_to_remove[0] + tolerance, color_to_remove[1] + tolerance, color_to_remove[2] + tolerance)
        
        #Conviertir a formato entero de 8 bits para la función inRange de OpenCV
        lower_bound = np.array([max(lower_bound[0], 0), max(lower_bound[1], 0), max(lower_bound[2], 0)], dtype=np.uint8)
        upper_bound = np.array([min(upper_bound[0], 179), min(upper_bound[1], 255), min(upper_bound[2], 255)], dtype=np.uint8)

        #Crear la mascara con los rasgos
        mask = cv2.inRange(img_colored, lower_bound, upper_bound)
        #Invertirla
        mask = cv2.bitwise_not(mask)

        #Se aplica una operación de apertura para eliminar el ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #Aplicar una operación de cierre para unir áreas de piel cercanas
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #Refinar la máscara mediante suavizado y eliminación de pequeños componentes
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        #Aplicar la mascara al fondo
        self.img_draw = cv2.bitwise_and(self.img, self.img, mask=mask)

        return self.img_draw
    
    '''Metodo para buscar el color predominate (hsv)'''
    def _find_dominant_color_hsv(self):      
        # Reshape the image to be a list of pixels
        pixels = self.img.reshape(-1, 3)
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
    def _find_dominant_color_lab(self):      
        # Reshape the image to be a list of pixels
        pixels = self.img.reshape(-1, 3)
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
    