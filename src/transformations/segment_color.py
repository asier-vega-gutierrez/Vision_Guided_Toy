import cv2
import numpy as np

'''Clase para segmentar por colores usando color lab'''
class SegmentColor:
    
    def __init__(self):
        self.img = None
        self.lab = None

    '''Mascara de color azul'''
    def _get_blue_shapes(self):
        lower = np.array([80, -50, -30])
        upper = np.array([255, 185, 120])
        mask = cv2.inRange(self.lab, lower, upper)
        return self._delete_noise(self.img, mask)

    '''Mascara de color verde'''
    def _get_green_shapes(self):
        lower = np.array([0, -100, 0])
        upper = np.array([200, 120, 180])
        mask = cv2.inRange(self.lab, lower, upper)
        return self._delete_noise(self.img, mask)
    
    '''Mascara de color rojo'''
    def _get_red_shapes(self):
        lower = np.array([0, 150, -255])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(self.lab, lower, upper)
        return self._delete_noise(self.img, mask)

    '''Mascara de color amarillo'''
    def _get_yellow_shapes(self):
        lower = np.array([185, 100, 140])
        upper = np.array([255, 120, 255])
        mask = cv2.inRange(self.lab, lower, upper)
        return self._delete_noise(self.img, mask)  
    
    '''Metodo para eliminar ruido y mejorar la deteccion del la fomra con color uniforme'''
    def _delete_noise(self, img, mask, soft=True):
        if soft:
            # Se aplica una operación de apertura para eliminar el ruido
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # Aplicar una operación de cierre para unir áreas de piel cercanas
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # Refinar la máscara mediante suavizado y eliminación de pequeños componentes
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        #Se aplica la mascara a la imagen y se obtiene el resultado
        return cv2.bitwise_and(img, img, mask=mask)
    
    '''Metodo principal para optener la segmetacion por colores con los filtros definidos'''
    def get_segmented_imgs(self, img):
        self.img = img
        #Se necesita la imagen en Color Lab ya que los colores se ven mejor
        self.lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        #Se aplican los filtros
        b = self._get_blue_shapes()
        g = self._get_green_shapes()
        r = self._get_red_shapes()
        y = self._get_yellow_shapes()
        #Se devuelven una imagen para cada color
        return {'blue': b, 'green': g, 'red': r, 'yellow': y}