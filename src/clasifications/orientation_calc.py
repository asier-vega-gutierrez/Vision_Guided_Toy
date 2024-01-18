import cv2
import numpy as np

'''Clase apra calcular la oritneacion del tablero y de las formas'''
class OrientationCalc:

    def __init__(self) -> None:
        self.img_draw = None
    
    '''Metodo para calcular la orientacion del tablero (un cuadrado)'''
    def calc_board_orientation(self, img, contour):
        
        #Obtener el rectángulo rotado alrededor del contorno
        rec = cv2.minAreaRect(contour)
        #Convertir las coordenadas del rectángulo a enteros
        box = np.int0(cv2.boxPoints(rec))
        #Linea del tablero y linea de la imagen (simpre recta verticalmente)
        board_line = [box[0], box[1]]
        straight_line = np.array([[0,0] , [0,720]])
        #Con las dos lineas podemos sacar un angulo
        degrees = self._cal_angle_2_lines(straight_line, board_line)
        #Dibujar la informacion
        self.img_draw = self._draw_img(img, box, board_line, straight_line)
        #Devolver el angulo del tablero
        return degrees, self.img_draw
        
    '''metod para iterar por todas las formas e ir completando su dato de angulo'''
    def calc_shape_orientation(self, img, data_shapes):
        #La linea recta de referencia
        straight_line = np.array([[0,0] , [0,720]])
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
                shape['angle'] = self._cal_angle_2_lines(straight_line, shape_line)
                #Pintamos la informacion en la imagen
                self._draw_img(img, box, shape_line, straight_line)
                
            elif shape['label'] == "Triangle":
                #Obtenemos el triangulos mas peuqeño que quepa en en controno
                triangle_contour = cv2.minEnclosingTriangle(shape['contour'])[1]
                #Buscamos la base del trinagulo (es quilatero las bases son iguales)
                base_start, base_end = triangle_contour[:2].astype(int)
                #Sacamos uno de los lados del triagulo
                top_point = triangle_contour[2]
                shape_line = [base_start, base_end]
                shape_line = np.array([shape_line[0][0], shape_line[1][0]])
                #Con las dos lineas podemos sacar un angulo
                shape['angle'] = self._cal_angle_2_lines(straight_line, shape_line)
                #Pintamos la informacion en la imagen
                self._draw_img(img, np.int0(triangle_contour), shape_line, straight_line)
                

            elif shape['label'] == "Hexagon":
                
                pass

            data_shapes[id] = shape

        for data in data_shapes:
            print(data['label'], data['center'], data['color'], data['merged'], data['angle'])
        cv2.imshow("Detected Lines", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None
       

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

    def _draw_img(self, img, box, board_line, straight_line):
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        cv2.line(img, board_line[0], board_line[1], (255,0,0), 3)
        cv2.line(img, straight_line[0], straight_line[1], (0,0,255), 3)
        return img