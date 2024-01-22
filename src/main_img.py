import cv2
from board.board import Board
from screen.display_3x3 import display3x3
from transformations.erase_background import EraseBackground
from transformations.segment_color import SegmentColor
from transformations.contours_cropper import ControursCropper
from transformations.pixel_2_real import Pixels2Real
from clasifications.find_board import FindBoard
from clasifications.shape_clasifier import ShapeClassifier
from clasifications.orientation_calc import OrientationCalc
from clasifications.movement_creation import MovementCreation
from data.data_transformation import DataTransformation


def main():

    #Objetos
    background_eraser = EraseBackground()
    color_segmenter = SegmentColor() 
    contour_cropper = ControursCropper()
    board_finder = FindBoard()
    shape_classifier = ShapeClassifier()
    data_transformer = DataTransformation()
    orientation_calculer = OrientationCalc()
    movement_creator = MovementCreation()
    #Listas
    out_data_shapes = []
    out_img_shapes = []
    in_data_shapes = []
    places_board_imgs = []
    list_movements = []
    list_movements_imgs = []

    #Adqusicion de la imagen
    img = cv2.imread("./data/test/imagen_1.jpg")
    img_conf = cv2.imread("./data/conf/conf_2.jpg")
    img_original = img.copy()

    #Base del tablero

    #Deteccion del tablero (Resultado: img con solo el tablero, img con todo menos el tablero)
    img_board, img_no_board, board_center, board_contour = board_finder.find_board(img)

    #Busqueda de formas en el tablero
    data_shape, detected, img_shape, in_contours = shape_classifier.classify_shapes(img_board, min_polygon_area = 2000, max_polygon_area = 15000, findcontours_type = cv2.RETR_TREE)
    if detected == True:
        for shape in data_shape:
            in_data_shapes.append(shape)
        img_shape_board = img_shape
        

    #Eliminado del fondo de la imagen sin tablero
    img_no_backgound = background_eraser.erase_background(img_no_board, color_type = 0, tolerance = 50)

    #Base de las formas

    #Busqueda de formas fuera del tablero por colores
    imgs_colors = color_segmenter.get_segmented_imgs(img_no_backgound)
    for id, img_color in imgs_colors.items():
        data_shape, detected, img_shape, out_contours = shape_classifier.classify_shapes(img_color, min_polygon_area = 2000, max_polygon_area = 15000, findcontours_type = cv2.RETR_EXTERNAL)
        if detected == True:
            for shape in data_shape:
                shape['color'] = id
                out_data_shapes.append(shape)
            out_img_shapes.append(img_shape)

    #Eliminado de la formas repitas por muy pocos pixeles de distancia fuera del tablero 
    out_data_shapes = data_transformer.filter_close_centers(out_data_shapes, 20)
    #Eliminado de la formas repitas por muy pocos pixeles de distancia en el tablero    
    in_data_shapes = data_transformer.filter_close_centers(in_data_shapes, 20)

    #Deteccion de huecos vacios o llenos en el tablero

    #Configurar el tablero
    img_board_conf, img_no_board_conf, board_center_conf, board_contour_conf = board_finder.find_board(img_conf) #se busca el tablero
    board_data_shapes_conf = [] #se clasifican las formas del tablero
    data_shape_conf, detected_conf, img_shape_conf, in_contours_conf = shape_classifier.classify_shapes(img_board_conf, min_polygon_area = 2000, max_polygon_area = 15000, findcontours_type = cv2.RETR_TREE)
    if detected_conf == True:
        for shape in data_shape_conf:
            board_data_shapes_conf.append(shape)
        img_shapes_board_conf = img_shape_conf
    board_data_shapes_conf = data_transformer.filter_close_centers(board_data_shapes_conf, 20) #Se limpian las formas del tablero

    #Creacion del tablero
    my_board = Board()
    #En funcion de la infomacion de la configuracion configuramos el tablero
    my_board.configure_board(board_data_shapes_conf, img_board_conf, board_center_conf)

    #Segemetacion de todos los huecos del tablero (imagenes separadas de las posciones encontrads en el tablero)
    places_board_imgs = contour_cropper.crop_contours_shapes(img, in_data_shapes, min_contour_area = 2000, max_contour_area = 15000)
    #Actualizar el tablero (las posiciones de in_data_shapes coinciden con las de places_board_imgs)
    my_board.update_board(in_data_shapes, places_board_imgs)
    print("Tablero actual:")
    for row in my_board.board_places:
        print(row)

    #Calculo de la orientacion

    #Calculo de la orientacion del tablero
    board_angle, img_orientation = orientation_calculer.calc_board_orientation(img.copy(), board_contour, in_data_shapes)
    #Calculo de la orientacion de las piezas
    out_data_shapes = orientation_calculer.calc_shape_orientation(img_orientation, out_data_shapes)

    #Creador de movimientos formas a tablero

    #Llamando a este metodo se genera la lsita de intrucciones para colocar las formas en el tablero
    list_movements, list_movements_imgs = movement_creator.create_movements(my_board, out_data_shapes, in_data_shapes, img.copy())

    #Estabelcimento del scalado pixel real

    #Configruacion de la imagen de referencia
    real_size_mm = 85
    #Deteccion de imagen de refencia y obtencion de escala
    pixel_converter = Pixels2Real(img_conf, real_size_mm)
    #Conversion de los datos
    for id, movement in enumerate(list_movements):
        movement['center'], img_distances = pixel_converter.pixel_2_real_point(movement['center'], img_conf.copy())
        movement['aim_center'], img_distances = pixel_converter.pixel_2_real_point(movement['aim_center'], img_conf.copy())
        movement['center'] = (round(movement['center'][0], 3), round(movement['center'][1], 3))
        movement['aim_center'] = (round(movement['aim_center'][0], 3), round(movement['aim_center'][1], 3))
        movement['x'] = round(pixel_converter.pixel_2_real_ditance(movement['x']), 3)
        movement['y'] = round(pixel_converter.pixel_2_real_ditance(movement['y']), 3)
        movement['angle'] = round(movement['angle'], 3)

        list_movements[id] = movement

    #Pintado final de la realizacion del movimiento
        
    #Listado de movimientos
    print("Lista de movimientos:")
    for movement in list_movements:
        print("Pieza: {:<10} Color: {:<10} Ubicada: {:<20} Angulo: {:<10} Objetivo: {:<20} x: {:<10} y: {:<10} Angulo objetivo: {:<10}".format(
            str(movement['label']), str(movement['color']), str(movement['center']), str(movement['angle']),
            str(movement['aim_center']), str(movement['x']), str(movement['y']), str(movement['aim_angle'])
        ))

    #Imagenes
    display3x3('imagen', img, 1)
    cv2.imwrite(f'./img/results/0-img.jpg', img)
    display3x3('imagen_conf', img_conf, 2)
    cv2.imwrite(f'./img/results/1-img_conf.jpg', img_conf)
    display3x3('imagen_conf_res', img_shape_conf, 3)
    cv2.imwrite(f'./img/results/2-img_shape_conf.jpg', img_shape_conf)
    display3x3('imagen_board', img_shape_board, 4)
    cv2.imwrite(f'./img/results/3-img_shapes_board.jpg', img_shape_board)
    display3x3('imagen_board_places', my_board.img_draw, 5)
    cv2.imwrite(f'./img/results/4-img_board_places.jpg', my_board.img_draw)
    for id, img in enumerate(out_img_shapes):
        display3x3(id, img, id+6)
        cv2.imwrite(f'./img/results/5-img_shape_{id}.jpg', img)
    for id, img in enumerate(list_movements_imgs):
        cv2.imshow('Result' + str(id), img)
        cv2.imwrite(f'./img/results/6-img_result_{id}.jpg', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()