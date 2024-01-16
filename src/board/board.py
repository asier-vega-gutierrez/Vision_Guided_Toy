from transformations.segment_color import SegmentColor
from data.data_transformation import DataTransformation
import cv2
import math
import numpy as np

class Board:

    def __init__(self) -> None:
        self.img_draw = None
        #suponer que si no la detecta ya esta relleno
        self.board_places = [[True, True, True, True],
                            [True, True, True, True],
                            [True, True, True, True],
                            [True, True, True, True]]
        #Esto se rellena al configurar le tablero con las posiciones de cada uno de los sitios detectados, (0,0) si no se ha podido encotrar sitio coincidente al al actualizar
        self.board_places_centers = [[(0,0), (0,0), (0,0), (0,0)],
                            [(0,0), (0,0), (0,0), (0,0)],
                            [(0,0), (0,0), (0,0), (0,0)],
                            [(0,0), (0,0), (0,0), (0,0)]]

    '''Metodo para configurar el tablero, necesita sus formas el tabelro y el centro del tabelro'''
    def configure_board(self, shapes, img, main_center):
        #Separa los centros
        centers = []
        for shape in shapes:
            centers.append(shape['center'])

        #Compovar que sean 16 centros (si no lo son se busca y elimina el centro del centro del tabelero)
        print("Numero de posicones encontradas durante la conf: " + str(len(centers)))
        if len(centers) != 16:
            #Eliminar el dentro del tablero, introducioendolo a la lista y luego borrando ambos
            centers.append(main_center)
            filtered_centers = []
            #Se busca el punto mas cercano al centro y luego se borran ambos
            for i in range(len(centers)):
                center_i = centers[i]
                is_far_enough = all(self.euclidean_distance(center_i, centers[j]) >= 20 for j in range(len(centers)) if i != j)
                if is_far_enough:
                    filtered_centers.append(centers[i])
            centers = filtered_centers
        #Pintar los puntos en la imagen
        self.img_draw = self._draw_help(centers, img)

        #Ordenar los puntos segun x mas pequeña primero
        sorted_centers = sorted(centers, key=lambda center: center[0])
        #introducir los centros ordenados
        row_count = len(self.board_places_centers)
        col_count = len(self.board_places_centers[0])
        center_index = 0
        for i in range(row_count):
            for j in range(col_count):
                self.board_places_centers[i][j] = sorted_centers[center_index]
                center_index += 1
        #Ordenar los puntos segun y mas pequeña primero por fila
        for id, row in enumerate(self.board_places_centers):
            sorted_row = sorted(row, key=lambda center: center[1])
            self.board_places_centers[id] = sorted_row  
                

    def update_board(self, data_shapes, img_places):
        color_segmenter = SegmentColor()
        data_transformer = DataTransformation()
        print("Numero de imagenes detectadas en el tablero: " + str(len(img_places))) #El nuemro de images deveria ser 16
        #Buscar las posiciones vacias
        data_shapes_empty = []
        for id, img in enumerate(img_places):
            #Se segmenta por colores (ahora se trabaja con img_places*4)
            imgs_colors = color_segmenter.get_segmented_imgs(img)
            #Por cada imagen con color se busca el color dominante
            for id_color, img_color in imgs_colors.items():
                dominant_color = self._find_dominant_color_hsv(img_color)
                #Si el color dominate es negro entronces el espacio esta vacio
                if dominant_color[0] < 100 or dominant_color[1] < 100 or dominant_color[2] < 100:  #TODO ESTO PUEDE NECESITAR AJUSTES SEGUN EL COLOR
                    #Como los ids (data_shapes y img_places) coinciden puedo almacenar los datos de los que se que estan vacios
                    data_shapes_empty.append(data_shapes[id])
                    print(data_shapes[id]['center'])



        #Ahora tengo cutatro de cada (uno por color), pero los que si ha detectado color van a tener uno menos
        #Circle (1033, 431) no yes (son tres de estos)
        #Hay que eliminar el grupo entero



        data_shapes_empty = data_transformer.filter_close_centers(data_shapes_empty, 20)
        for data in data_shapes_empty:
            print(data['label'], data['center'], data['color'], data['merged'])
        pass

        #Haora hay que buscar el centro que mas se parezca a los almacenados y ponerlo a false y el resto a true sea lo que sea

    '''Metodo para sacar la distancia euclidea entre dos puntos'''
    def euclidean_distance(self, center1, center2):
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    

    def _draw_help(self, centers, img):
        #pintar los centros
        for center in centers:
            if center is not None:
                cv2.circle(img, center, 5, (255, 255, 255), -1)
        return img
            
    '''Metodo para buscar el color predominate (hsv)'''
    def _find_dominant_color_hsv(self, img):      
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