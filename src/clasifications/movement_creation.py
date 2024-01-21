import cv2

'''Metodo para crear los movimientos segun informacion extraida'''
class MovementCreation:

    def __init__(self) -> None:
        self.list_movements = []
        self.imgs_draw = []

    '''Este metodo recive la informacion del tabelerlo, las formas de fuera y de dentro del tabelero'''
    def create_movements(self, board, out_data_shapes, in_data_shapes, img):

        #Se añade un atributo find al diccionario de las fomas de fuera
        for id, shape in enumerate(out_data_shapes):
            shape['find'] = False
            out_data_shapes[id] = shape

        #BIEN
        for data in out_data_shapes:
            print(data['label'], data['center'], data['color'], data['merged'], data['angle'])
        for data in in_data_shapes: 
            print(data['label'], data['center'], data['color'], data['merged'], data['angle'])
            
        #Se itera por todas las formas encontradas fuera junto con todas las posciones encrontradas dentro
        for id, out_shape in enumerate(out_data_shapes):
            for in_shape in in_data_shapes:
                #Se comprueba que la label de ambas coincida y que esa forma aun no tenga sitio asignado
                if out_shape['label'] == in_shape['label'] and out_shape['find'] == False:
                    #Para saber que sitio se le asigna a la pieza de fuera se ite por los tipos de formas del tablero 
                    for id_row, row in enumerate(board.board_places_type):
                        for id_place, place in enumerate(row):
                            #Se comprueba que la label concida con los valoras predefinidos (la primera comprovacion peude ser erronea), que ese sitio este libre y que esa forma aun no tenga sitio asignado
                            if out_shape['label'] == place and board.board_places[id_row][id_place] == False and out_shape['find'] == False:
                                #Con esto ya tendriamos la pieza colacada, ahora se guardan las cordenadas obejtivo
                                aim_cordinates = board.board_places_centers[id_row][id_place]

                                #EL PROBLEMA ESTA EN board_places_type NO SE MUY BIEN PROQUE
                                print(out_shape['label'] + board.board_places_type[id_row][id_place])
                                cv2.circle(img, aim_cordinates, 4, (255,0,0))
                                cv2.circle(img, out_shape['center'], 4, (255,0,0))
                                cv2.imshow('Image', img)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()

                                #Tambien es necesario establecer un angulo objetivo (como se coje de referencia el tablero se peuden poner talcual)
                                if out_shape['label'] == "Square":
                                    aim_angle = self._find_nearest_angle_square(out_shape['angle'])
                                if out_shape['label'] == "Triangle":
                                    aim_angle = 270
                                if out_shape['label'] == "Hexagon":
                                    aim_angle = 90
                                if out_shape['label'] == "Circle":
                                    aim_angle = 0
                                #Y se fija la posicion como ocupada
                                board.board_places[id_row][id_place] = True 
                                #Se necesita actualizar el valor de la pieza encotrada para que no vuelva a iterarse en el bucle prncipal
                                out_shape['find'] = True
                                out_data_shapes[id] = out_shape

                                #Añadimos el movimiento a la lista junto con el resto de datos
                                self.list_movements.append({'label': out_shape['label'], 
                                                            'center': out_shape['center'],
                                                            'aim_center': aim_cordinates,
                                                            'x': self._calc_distance(out_shape['center'], aim_cordinates, 0),
                                                            'y': self._calc_distance(out_shape['center'], aim_cordinates, 1),
                                                            'angle': out_shape['angle'],
                                                            'aim_angle': aim_angle,
                                                            'color': out_shape['color'], 
                                                            'out_merged': out_shape['merged'], 
                                                            'in_merged': in_shape['merged'],
                                                            'out_contour': out_shape['contour'],
                                                            'in_contour': in_shape['contour']})
                            
        for movement in self.list_movements:
            self.imgs_draw.append(self._draw_help(img.copy(), movement['center'], movement['aim_center'], movement['x'], movement['y']))
        
        return self.list_movements, self.imgs_draw

    '''Metodo para obetener la distacia entre pieza fuera y hueco dentro'''
    def _calc_distance(self, out_center, in_center, cord):
        if out_center[cord] < in_center[cord]:
            return in_center[cord] - out_center[cord] 
        else:
            return out_center[cord] - in_center[cord] 
    
    '''Metodo para bucar el angulo mas cercano para colocar un cuadrado'''
    def _find_nearest_angle_square(self, angle):
        angles = [0, 90, 180, 270]
        closest_angle = min(angles, key=lambda x: abs(x - angle))
        return closest_angle
    
    '''Metodo para pinta la iamgen'''
    def _draw_help(self, img, out_center, in_center, x, y):
        cv2.arrowedLine(img, out_center, in_center, (255,0,0), 3, tipLength=0.05)
        if out_center[0] < in_center[0]:
            cv2.line(img, out_center, (out_center[0] + x, out_center[1]), (0,0,255), 2)
            if out_center[1] < in_center[1]: 
                cv2.line(img, (out_center[0] + x, out_center[1]), (out_center[0] + x, out_center[1] + y), (0,0,255), 2)
            else:
                cv2.line(img, (out_center[0] + x, out_center[1]), (out_center[0] + x, out_center[1] - y), (0,0,255), 2)
        else:
            cv2.line(img, out_center, (out_center[0] - x, out_center[1]), (0,0,255), 2)
            if out_center[1] < in_center[1]: 
                cv2.line(img, (out_center[0] - x, out_center[1]), (out_center[0] - x, out_center[1] + y), (0,0,255), 2)
            else:
                cv2.line(img, (out_center[0] - x, out_center[1]), (out_center[0] - x, out_center[1] - y), (0,0,255), 2)
        return img