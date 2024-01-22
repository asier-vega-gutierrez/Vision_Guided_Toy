import cv2
import pickle
from typing import Dict, Any


class Camera:
    '''Se necesita el id que windows da a la camara, la velocidad de grabado, y la resolucion de la camara'''
    def __init__(self, hw_id: int, w: int, h: int):
        #Recording parameters
        self.video_capturer = self.set_video_capute(hw_id, w, h) #Capturador de video configurado segun hardware
        self.size = (int(self.video_capturer.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_capturer.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame = None #Frame actual capturado
        self.success = None #Estado de la carpura del frame TODO
        #Calibration parameters
        self.camera_matrix = self.open_file_parameters("cameraMatrix.pkl") #Matriz resultante de la calibracion
        self.dist_coeffs = self.open_file_parameters("dist.pkl") #Coneficiontes de las distancias

    '''Metodo para configurar los parametros de la camara'''
    def set_video_capute(self, hw_id: int, w: int, h: int):
        cap = cv2.VideoCapture(hw_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w) #ancho de la imagen
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h) #alto de la imagen
        return cap

    '''Metodo para recuperar el frame actual de la camara'''
    def get_frame(self):
        self.record()
        return self.frame
    
    '''Metodo para mostrar el frame actual de la camara'''
    def show_frame(self):
        self.record()
        cv2.imshow('Camera_streaming', self.frame)
        cv2.waitKey(1)

    '''Metodo para gravar segun configuracion de la calibracionS'''
    def record(self):
        #Leer el frame
        success, frame = self.video_capturer.read()
        if success:
            #Recupear la are de interes segun matriz optima
            h,  w = frame.shape[:2]
            newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
            #Unidistorsion + remapeo
            mapx, mapy = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, newCameraMatrix, (w,h), 5)
            dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            #Cropeado de la imagen segun resultados
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            #Seteado del frame actual
            self.frame = dst
        else:
            print("E : error al capturar el frame")
            return


    '''Metodo para liberar la camara'''
    def release(self):
        self.video_capturer.release()
        cv2.destroyAllWindows()

    '''Metodo para leer los archvios de guardado con los parametros'''
    def open_file_parameters(self, file: str):
        with open('./src/camera/calibration/parameters/' + file, 'rb') as f:
            calibration_data = pickle.load(f)
        return calibration_data
