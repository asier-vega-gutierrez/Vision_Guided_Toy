import math
from common.common import euclidean_distance

class DataTransformation:
  
    '''Metodo para eliminar los centros cencanos, los eliminamos se les saca la media y se vuelve a introducir'''
    def filter_close_centers(self, shapes, distance_threshold):

        filtered_shapes = []
        descarted_shapes = []
        merged_shapes = []

        #Primero descartamos todas las figuras que esten muy cerca
        for i in range(len(shapes)):
            center_i = shapes[i]['center']
            is_far_enough = all(euclidean_distance(center_i, shapes[j]['center']) >= distance_threshold for j in range(len(shapes)) if i != j)
            if is_far_enough:
                filtered_shapes.append(shapes[i])
            else:
                descarted_shapes.append(shapes[i])

        #Si ha descatadas las adapatamos para meterlas (una por lo menos deve entrar)
        if descarted_shapes:  
            #Agrupamos las formas segun su distacia 
            grouped_shapes = {}
            for i in range(len(descarted_shapes)):
                shape_i = descarted_shapes[i]
                center_i = shape_i['center']
                #Se va comprovando que no haya un grupo cerca ya creado
                is_close_to_group = False
                for center, shapes_in_group in grouped_shapes.items():
                    if all(euclidean_distance(center_i, shape['center']) < distance_threshold for shape in shapes_in_group):
                        grouped_shapes[center].append(shape_i)
                        is_close_to_group = True
                        break
                #Si no se encutra grupo se crea uno nuevo con esa forma
                if not is_close_to_group:
                    grouped_shapes[center_i] = [shape_i]
            #Para cada grupo se calcula la media del centro
            merged_shapes = []
            for center, shapes in grouped_shapes.items():
                mean_center = (
                    int(sum(x for x, _ in [s['center'] for s in shapes]) / len(shapes)),
                    int(sum(y for _, y in [s['center'] for s in shapes]) / len(shapes))
                )
                #Señade un registro por cada grupo creado
                merged_shape = {
                    'label': shapes[0]['label'], #Se coje la primera label del grupo
                    'center': mean_center,
                    'color': shapes[0]['color'], #Se coje el primer color del grupo
                    'merged': 'yes', #Se marca como merged
                    'contour': shapes[0]['contour'], #Se coje el primer contorno del grupo
                    'angle': 0
                }
                merged_shapes.append(merged_shape)
        #Añadimos los registros nuevos 
        for shape in merged_shapes:
            filtered_shapes.append(shape)
        #Devolvemos la lsita con todas los registros
        return filtered_shapes
    
    '''Metodo para sacar la distancia euclidea entre dos puntos'''
    '''def euclidean_distance(self, center1, center2):
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)'''