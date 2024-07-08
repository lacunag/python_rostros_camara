import cv2

# Cargamos el clasificador preentrenado para detección de caras
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargamos la imagen en la que queremos detectar caras
image = cv2.imread('imagen.jpg')

# Convertimos la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectamos las caras en la imagen
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

# Dibujamos un rectángulo alrededor de cada cara detectada
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Mostramos la imagen con las caras detectadas
cv2.imshow('Faces detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
