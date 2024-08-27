from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("yolov8s.pt")

# Ruta de la imagen
image_path = "image.jpg"

# Leer la imagen
frame = cv2.imread(image_path)

# Leemos resultados
resultados = model.predict(frame, imgsz=640, conf=0.40)

# Mostramos resultados
anotaciones = resultados[0].plot()

# Mostramos nuestra imagen
cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

# Esperar por cualquier tecla y luego cerrar la ventana
cv2.waitKey(0)
cv2.destroyAllWindows()