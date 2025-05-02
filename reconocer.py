import face_recognition as fr
import cv2
import os

# Base de imágenes
images_dir = './base_imagenes/lfw_funneled'
test_image_path = './prueba.jpg'

# Datos codificados
face_encodings_db = []
face_names_db = []

def cargar_base_datos(images_dir):
    for subfolder in os.listdir(images_dir):
        subfolder_path = os.path.join(images_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    ruta = os.path.join(subfolder_path, filename)
                    nombre = subfolder  # Usamos el nombre del folder como etiqueta
                    imagen = fr.load_image_file(ruta)
                    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                    try:
                        encoding = fr.face_encodings(imagen_rgb)[0]
                        face_encodings_db.append(encoding)
                        face_names_db.append(nombre)
                        print(f"✅ Agregado: {nombre} ({filename})")
                    except IndexError:
                        print(f"❌ No se detectó rostro en: {filename}")

# Cargar base
cargar_base_datos(images_dir)

# Imagen de prueba
img_prueba = fr.load_image_file(test_image_path)
img_prueba_rgb = cv2.cvtColor(img_prueba, cv2.COLOR_BGR2RGB)

# Detección
face_locations = fr.face_locations(img_prueba_rgb)
face_encodings = fr.face_encodings(img_prueba_rgb, face_locations)

for encoding, location in zip(face_encodings, face_locations):
    resultados = fr.compare_faces(face_encodings_db, encoding, tolerance=0.5)
    distancias = fr.face_distance(face_encodings_db, encoding)

    mejor_match = None
    if len(distancias) > 0:
        idx_mejor = distancias.argmin()
        if resultados[idx_mejor]:
            mejor_match = face_names_db[idx_mejor]

    top, right, bottom, left = location
    cv2.rectangle(img_prueba_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
    nombre = mejor_match if mejor_match else "Desconocido"
    cv2.putText(img_prueba_rgb, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Mostrar resultado
cv2.imshow("Reconocimiento Facial", cv2.cvtColor(img_prueba_rgb, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
