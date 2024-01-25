import cv2
import numpy as np

'''Clase apra calcular la oritneacion del tablero y de las formas'''
class OrientationCalc:

    def __init__(self) -> None:
        self.img_draw = None
        self.straight_line = None
    
    '''Metodo para calcular la orientacion del tablero (un cuadrado)'''
    def calc_board_orientation(self, img, contour, data_shape):
        #Obtener el rectángulo rotado alrededor del contorno
        rec = cv2.minAreaRect(contour)
        #Convertir las coordenadas del rectángulo a enteros
        box = np.int0(cv2.boxPoints(rec))
        #Linea del tablero (el tablero siempre deve tener el cuadrado de la esquina como la forma mas cercana a 0,0)
        centers = []
        for shape in data_shape:
            centers.append(shape['center'])
        nearest_center = self._find_nearest_point(centers)
        #La forma de la esquina mas cercana a 0,0 del tablero deve ser un cudrado 
        corner_shape = None
        for shape in data_shape:
            if nearest_center == shape['center']:
                corner_shape = shape
        if corner_shape['label'] == "Square":
            board_line = [box[0], box[1]] #en el caso de qu los requisitos se cumplan se establece esta linea como la principal
        else:
            raise Exception("la esquina deve ser un cudrado")
        #La linea de la imagen para sacar el agulo con respecto a la camara (no se usa)
        straight_line = np.array([[0,0] , [0,720]])
        #Con las dos lineas podemos sacar un angulo
        degrees = self._cal_angle_2_lines(straight_line, board_line)
        #Dibujar la informacion
        self.img_draw = self._draw_img(img, box, board_line, straight_line)
        #Guardo mi linea del tablero como linea recta para despues usarla con las formas
        self.straight_line = board_line
        #Devolver el angulo del tablero
        return degrees, self.img_draw 
        
    '''metod para iterar por todas las formas e ir completando su dato de angulo'''
    def calc_shape_orientation(self, img, data_shapes):
        #Hya que comprobar que forma es y segun eso hacer una caso distinta
        for id, shape in enumerate(data_shapes):
            if shape['label'] == "Square":
                #Obtener el rectángulo rotado alrededor del contorno
                rec = cv2.minAreaRect(shape['contour'])
                #Convertir las coordenadas del rectángulo a enteros
                box = np.int0(cv2.boxPoints(rec))
                #Linea del cuadrado
                shape_line = [box[0], box[1]]
                #Con las dos lineas podemos sacar un angulo
                shape['angle'] = self._cal_angle_2_lines(self.straight_line, shape_line)
                #Pintamos la informacion en la imagen
                self._draw_img(img, box, shape_line, self.straight_line)
                
            elif shape['label'] == "Triangle":
                #Obtenemos el triangulos mas peuqeño que quepa en en controno
                triangle_contour = cv2.minEnclosingTriangle(shape['contour'])[1]
                #Buscamos la base del trinagulo (es quilatero las bases son iguales)
                base_start, base_end = triangle_contour[:2].astype(int)
                #Sacamos uno de los lados del triagulo
                shape_line = [base_start, base_end]
                shape_line = np.array([shape_line[0][0], shape_line[1][0]])
                #Con las dos lineas podemos sacar un angulo
                shape['angle'] = self._cal_angle_2_lines(self.straight_line, shape_line)
                #Pintamos la informacion en la imagen
                self._draw_img(img, np.int0(triangle_contour), shape_line, self.straight_line)
                
            elif shape['label'] == "Hexagon":
                # Assuming the contour of the hexagon is available in shape['contour']
                hexagon_contour = shape['contour']
                # Approximate the hexagon from the contour
                epsilon = 0.02 * cv2.arcLength(hexagon_contour, True)
                approx_hexagon = cv2.approxPolyDP(hexagon_contour, epsilon, True)
                # Convert the datatype of the points to integers
                approx_hexagon = approx_hexagon.astype(int)
                # Line of the hexagon
                shape_line = [approx_hexagon[0], approx_hexagon[1]]
                shape_line = np.array([shape_line[0][0], shape_line[1][0]])
                # Calculate the angle between the reference straight line and the hexagon's line
                shape['angle'] = self._cal_angle_2_lines(self.straight_line, shape_line)
                # Draw the hexagon and lines on the image
                self._draw_img(img, approx_hexagon, shape_line, self.straight_line)
            #Metemos el angluo alcula a su respectiva forma en la lista
            data_shapes[id] = shape
        #Devolvemos la nueva lista con los angulos
        return data_shapes
       
    '''Metodo para calcular el angulo entre dos lieneas'''
    def _cal_angle_2_lines(self, straight, line):
        #Calcular el vector de dirección para cada línea
        vector_line = line[1] - line[0]
        vector_straight = straight[1] - straight[0]
        #Calcular el producto punto
        producto_punto = np.dot(vector_line, vector_straight)
        #Calcular las magnitudes de los vectores
        magnitud_board = np.linalg.norm(vector_line)
        magnitud_straight = np.linalg.norm(vector_straight)
        #Calcular el coseno del ángulo entre las dos líneas
        coseno_theta = producto_punto / (magnitud_board * magnitud_straight)
        #Calcular el ángulo en radianes
        theta_rad = np.arccos(np.clip(coseno_theta, -1.0, 1.0))
        #Convertir el ángulo a grados
        theta_degrees = np.degrees(theta_rad)
        return theta_degrees

    '''Metodo para buscar el punto mas cercano'''
    def _find_nearest_point(self, points):

        # Convert the list of points to a NumPy array for easier calculations
        points_array = np.array(points)

        # Calculate the distances from each point to (0,0)
        try:
            distances = np.linalg.norm(points_array, axis=1)
        except np.exceptions.AxisError:
            raise(ValueError("Error al calcular la orientacion"))

        # Find the index of the point with the minimum distance
        index_of_nearest_point = np.argmin(distances)

        # Get the nearest point
        nearest_point = tuple(points_array[index_of_nearest_point])

        return nearest_point


    def _draw_img(self, img, box, board_line, straight_line):
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        cv2.line(img, board_line[0], board_line[1], (255,0,0), 3)
        cv2.line(img, straight_line[0], straight_line[1], (0,0,255), 3)
        return img