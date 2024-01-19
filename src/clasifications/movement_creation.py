import cv2

class MovementCreation:

    def __init__(self) -> None:
        self.list_movements = None
        self.img_draw = None

    def create_movements(self, board, out_data_shapes, in_data_shapes, img):

        for data in out_data_shapes:
            print(data['label'], data['center'], data['color'], data['merged'], data['angle'])
        print("---")
        for data in in_data_shapes:
            print(data['label'], data['center'], data['color'], data['merged'], data['angle'])
        print("---")


        for out_shape in out_data_shapes:
            for in_shape in in_data_shapes:
                if out_shape['label'] == in_shape["label"]: # se me ocurre añadir un find a out_shspes y compovarlo aqui para que una ficha no rellene de mas sitios
                    find = False
                    #Donde se coloca
                    for id_row, row in enumerate(board.board_places_type):
                        for id_place, place in enumerate(row):
                            if out_shape['label'] == place and board.board_places[id_row][id_place] == False:
                                find = True
                                aim_cordinates = board.board_places_centers[id_row][id_place]
                                board.board_places[id_row][id_place] = True 

                                print(out_shape['label'], out_shape['center'], out_shape['color'], out_shape['merged'], out_shape['angle'])
                                print(board.board_places_centers[id_row][id_place])                               
                                self.img_draw = self._draw_help(img, out_shape['center'], aim_cordinates)
                                cv2.imshow('Image', self.img_draw)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()


                    #Cuanto hay que girarla 

    def _draw_help(self, img, out_center, in_center):
        cv2.circle(img, out_center, 5, (255, 0, 0), -1)
        cv2.circle(img, in_center, 5, (255, 0, 0), -1)
        return img