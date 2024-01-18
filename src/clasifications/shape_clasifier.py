import cv2
import numpy as np

'''Clase para clasificar las formas que se encunentre un imagen'''
class ShapeClassifier:
    
    def __init__(self) -> None:
        #Imagen a buscar la formas e imagen para pintar informacion util
        self.img =  None
        self.img_draw = None
        self.img_gray = None #TODO No necesito consulta este atributo borrar 
        self.img_blurred = None
        self.img_thresholded = None
        self.contours = None
        self.img_contours = None

        #Varaible para saber si se ha detectado una forma o no
        self.detected = False

        #Estas varaibles almacenan los bordes para cada tipo de forma
        self.square_cnts = []
        self.circle_cnts = []
        self.triangle_cnts = []
        self.hexagon_cnts = []
        self.hough_circles = []

        #Esta variable guarda la iformacion para cada foram detectaad Ej.: {'label': "Square", 'center': (100,100)}
        self.shape_info = []

    def reset_cnts(self):
        self.square_cnts = []
        self.circle_cnts = []
        self.triangle_cnts = []
        self.hexagon_cnts = []
        self.hough_circles = []
        self.shape_info = []

    '''Metodo para clasificar las formas'''
    def classify_shapes(self, img, min_polygon_area, max_polygon_area, findcontours_type):
        self.detected = False
        self.reset_cnts() #TODO

        self.img = img.copy()
        self.img_draw = img.copy()

        #Preprocesado
        self._preproces(findcontours_type)

        #Busqueda de formas basica, se itera por los contronos detectados
        for contour in self.contours:
            #Usamos la funcion approsPolyDP, para que reconozca poligonos regulares atraves de los contrornos detectados (deven estar cerrrados)
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            #sacamos el area del contorno
            area = cv2.contourArea(contour)
            #Solo si el area es muy pequeña se elimina, esto reduce le rudio que genera el filtro adaptiveThreshold()
            if area > min_polygon_area and area < max_polygon_area:
                #Buscamos el nuemero de lados
                sides = 0
                sides = len(approx)
                #En funcion de los lados que la funcion approxPolyDP nos devuelva clasificamos el controno y añadimo informacion relevante
                if sides == 4:
                    self.square_cnts.append(contour)
                    self.shape_info.append({'label': "Square", 'center': self._find_centrer(contour), 'color': 'no', 'merged': 'no', 'contour': contour, 'angle': 0})
                    self.detected = True
                elif sides == 3:
                    self.triangle_cnts.append(contour)
                    self.shape_info.append({'label': "Triangle", 'center': self._find_centrer(contour), 'color': 'no', 'merged': 'no', 'contour': contour, 'angle': 0})
                    self.detected = True
                elif sides == 6:
                    self.hexagon_cnts.append(contour)
                    self.shape_info.append({'label': "Hexagon", 'center': self._find_centrer(contour), 'color': 'no', 'merged': 'no', 'contour': contour, 'angle': 0})
                    self.detected = True
                else:
                    self.circle_cnts.append(contour)
                    self.shape_info.append({'label': "Circle", 'center': self._find_centrer(contour), 'color': 'no', 'merged': 'no', 'contour': contour, 'angle': 0})
                    self.detected = True

                #Busqueda de circulos por Hough, es mas preciso que el metodo anteriro en el caso de los circulos
                '''circles = cv2.HoughCircles(self.img_blurred,cv2.HOUGH_GRADIENT,1,120,param1=100,param2=20,minRadius=min_hough_radius,maxRadius=200)
                if circles is not None and len(circles) > 0:
                    self.hough_circles = np.uint16(np.around(circles))
                    for circle in self.hough_circles[0,:]:
                        self.shape_info.append({'label': "Circle", 'num_sides': 0, 'center': (circle[0],circle[1])})
                        self.detected = True'''

        #Si queremos una imagen con informacion pintada
        self.img_draw = self._draw_help(self._draw(self.img_draw))
        #Se devuelve la informacion, si se a detectado alguna forma y las imagenes pintadas con las formas
        return self.shape_info, self.detected, self.img_draw, self.contours

    '''Metodo de preprocesado para la deteccion de contornos '''
    def _preproces(self, findcontours_type):
        #Pasar la imagen a escala de grises
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        #Dar blurr a la imagen
        self.img_blurred = cv2.GaussianBlur(self.img_gray, (5, 5), 0)
        #Threshold adaptativo
        block_size = 5  # Size of the pixel neighborhood used for adaptive thresholding
        c = 3  # Constant subtracted from the mean
        self.img_thresholded = cv2.adaptiveThreshold(self.img_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, c)
        #Deteccion de bordes
        self.contours, hierarchy = cv2.findContours(self.img_thresholded, findcontours_type, cv2.CHAIN_APPROX_SIMPLE)
        self.img_contours = cv2.drawContours(self.img, self.contours, -1, (255,255,0), 2) 

    '''Metodo para encontra el centro de un poligono'''
    def _find_centrer(self, contour):
        #Se crea el circulo mas peqeuño que contenga al poligono, y se coje el centro del de ese ciculo
        (x, y), _ = cv2.minEnclosingCircle(contour)
        return (int(x), int(y))

    '''Metodo para pintar los poligonos clasificados'''
    def _draw(self, img):
        #Formas basicas
        cv2.drawContours(img, self.square_cnts, -1, (0, 255, 0), 2)
        cv2.drawContours(img, self.circle_cnts, -1, (0, 0, 255), 2)
        cv2.drawContours(img, self.triangle_cnts, -1, (255, 0, 0), 2)
        cv2.drawContours(img, self.hexagon_cnts, -1, (255, 255, 0), 2)
        #Centros de formas basicas
        for shape_data in self.shape_info:
            center = shape_data['center']
            if center is not None:
                cv2.circle(img, center, 2, (255, 255, 255), -1)
        #Ciruclos de hough y sus centros
        '''if len(self.hough_circles) != 0: 
            for circle in self.hough_circles[0,:]:
                cv2.circle(img,(circle[0],circle[1]),circle[2],(0, 0, 255),2)
                cv2.circle(img,(circle[0],circle[1]),2,(255, 255, 255),-1)'''
        return img

    '''Metodo que pinta ayuda en las iamgenes'''
    def _draw_help(self, img):
        #Se pinta el centro detectado y la label
        for shape_data in self.shape_info:
            center = shape_data['center']
            label = shape_data['label']
            if center is not None:
                cv2.putText(img, label, (int(center[0]) - 30, int(center[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
        return img