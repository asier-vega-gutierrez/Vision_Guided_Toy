import cv2

# Función para guardar la imagen actual con un nombre único
def guardar_imagen(frame, contador):
    nombre_archivo = f'imagen_{contador}.jpg'
    cv2.imwrite(nombre_archivo, frame)
    print(f'Imagen guardada como {nombre_archivo}')
    return contador + 1

# Abre la cámara (puedes cambiar el número 0 por la ruta de un archivo de video si lo deseas)
cap = cv2.VideoCapture(0)
# Resolucion
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Inicializa el contador
contador = 1

while True:
    # Captura el fotograma actual
    ret, frame = cap.read()

    # Muestra el fotograma en una ventana llamada "Frame"
    cv2.imshow('Frame', frame)

    # Tamaño de la imagen gravada
    #print(frame.shape)

    # Espera a que se presione una tecla
    key = cv2.waitKey(1) & 0xFF

    # Si la tecla presionada es "s", guarda la imagen actual y actualiza el contador
    if key == ord('s'):
        contador = guardar_imagen(frame, contador)
    # Si la tecla presionada es "q", sale del bucle
    elif key == ord('q'):
        break

# Libera la captura y cierra la ventana
cap.release()
cv2.destroyAllWindows()