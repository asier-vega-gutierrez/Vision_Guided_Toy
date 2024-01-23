import cv2
import numpy as np

'''Calse para cropear imagens segun contronos pasados'''
class ControursCropper:

        def __init__(self) -> None:
                self.img =  None
                self.cropped_image = []

        '''Metodo para cropear la imagen del tablero segun laos contornos pasados'''
        def crop_contours_shapes(self, img, data_shape, min_contour_area, max_contour_area):
                self.cropped_image = []
                self.img = img
                for shape in data_shape:
                        area = cv2.contourArea(shape['contour'])
                        if area > min_contour_area and area < max_contour_area:
                                x, y, w, h = cv2.boundingRect(shape['contour'])
                                self.cropped_image.append(img[y:y + h, x:x + w])
                return self.cropped_image
