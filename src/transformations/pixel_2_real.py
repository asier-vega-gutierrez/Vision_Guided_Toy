import cv2
from pyzbar.pyzbar import decode
import numpy as np

'''Clase para trasformar la informacion leida por la camara en real'''
class Pixels2Real:
    #De froma inicial se necesita una imagen con un codigo qr y el tamaño de ese qr en milimetros
    def __init__(self, img_reference, real_size_mm):
        self.img_reference = img_reference
        self.real_size_mm = real_size_mm
        #Aqui se iniciliza el numero para realiza rle cambio pixel a mm y el punto de cordenadas (0,0) 
        self.scale, self.cordinate_point = self._get_mm_per_pixel_scale_and_qr_corner()
        self.img_draw = None
        self.y_pixels = None
        self.x_pixels = None
        self.point_mm = None
    
    '''Llamando a este metodo se consigue pasar la informacion de un punto en pixeles a real'''
    def pixel_2_real(self, point, img):
        
        #Generamos la distacia en base al neuvo punto de cordenadas
        self.y_pixels = point[0] - self.cordinate_point[0]
        self.x_pixels = point[1] - self.cordinate_point[1]

        #Creamos el putno en mm
        self.point_mm = (self.y_pixels * self.scale, self.x_pixels * self.scale)

        #Si queremos una imagen con informacion pintada
        self.img_draw = img
        self.img_draw = self._draw_help(self._draw(self.img_draw))

        #Se devuelve el punto en mm y la imagen con la ayuda
        return self.point_mm, self.img_draw
    
    '''Metodo para pintar lineas de ayuda sobre la imagen'''
    def _draw(self, img):
        w, h, _ = img.shape

        #Centro de cordenads nuevo
        cv2.circle(img, self.cordinate_point, 3, (0, 0, 255), -1)
        cv2.line(img, self.cordinate_point, (self.cordinate_point[0], w), (0, 0, 255), 2)
        cv2.line(img, self.cordinate_point, (h, self.cordinate_point[1]), (0, 0, 255), 2)
        cv2.line(img, self.cordinate_point, (self.cordinate_point[0], 0), (0, 0, 255), 2)
        cv2.line(img, self.cordinate_point, (0, self.cordinate_point[1]), (0, 0, 255), 2)


        #Distancias
        cv2.line(img, (self.cordinate_point[0], self.x_pixels + self.cordinate_point[1]), (self.y_pixels + self.cordinate_point[0], self.x_pixels + self.cordinate_point[1]), (0, 255, 0), 2)
        cv2.line(img, (self.y_pixels + self.cordinate_point[0], self.cordinate_point[1]), (self.y_pixels + self.cordinate_point[0], self.x_pixels + self.cordinate_point[1]), (255, 0, 0), 2)
        
        return img

    '''Metodo para pintar texto de ayuda sobre la imangen'''
    def _draw_help(self, img):
        
        fuente = cv2.FONT_HERSHEY_SIMPLEX

        texto1 = "y: "+ str(round(self.point_mm[0], 2))
        posicion_texto1 = (self.cordinate_point[0], self.x_pixels + self.cordinate_point[1] - 10)
        cv2.putText(img, texto1, posicion_texto1, fuente, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        texto2 = "x: "+ str(round(self.point_mm[1], 2))
        posicion_texto2 = (self.y_pixels + self.cordinate_point[0] + 5, self.cordinate_point[1] + 15)
        cv2.putText(img, texto2, posicion_texto2, fuente, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
        return img
    
    '''metodo para obtener el porcentaje de escalado pixel a mm y el punto de cordenada inicial'''
    def _get_mm_per_pixel_scale_and_qr_corner(self):
        #Se necesita hacer un pequeño preporcesado para poder leer el qr
        img_preprocessed = self._preproces(self.img_reference)

        #Buscamos los qr de la iamgen, deveria solo encontrarse uno
        qr_codes = decode(img_preprocessed)

        if qr_codes:
            #Recuperamos el punto mas cercano al (0,0) de la imagen y la anchura del qr
            qr_code = qr_codes[0]
            (x, y, w, h) = qr_code.rect

            #Convertir las coordenadas del polígono a un formato adecuado para cv2.minAreaRect
            polygon_points = np.array(qr_code.polygon, dtype=np.int32)

            #Obtener el rectángulo delimitador mínimo (rotado) del QR
            rect = cv2.minAreaRect(polygon_points)

            #Calcular el centro del QR desde el rectángulo delimitador
            qr_corner_coordinates = (int(rect[0][0]), int(rect[0][1]))

            #Con la anchura se saca la escala de conversion
            mm_per_pixel_scale = self.real_size_mm / w

            #El punto srive como nuevo punto de cordenadas
            #qr_corner_coordinates = (x, y)
            #qr_corner_coordinates = (x + w // 2, y + h // 2)

            return mm_per_pixel_scale, qr_corner_coordinates
        
        else:
            print("No QR codes found in the image.")
            return None

    '''Preprocesado para la deteccion del qr'''
    def _preproces(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        return img_blurred