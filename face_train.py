# Библиотека компьютерного зрения
import cv2
# Библиотека для вызова системных функций
import os
# Библиотека для обучения нейросетей
import numpy as np
# Встроенная библиотека для работы с изображениями
from PIL import Image

# Получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))
# Создаём новый распознаватель лиц
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Указываем, что мы будем искать лица по примитивам Хаара
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# Путь к датасету с фотографиями пользователей
dataPath = path + r'/dataSet'

# Получаем картинки и id из датасета
def get_images_and_labels(datapath):
     # Получаем путь к картинкам
     image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
     # Списки картинок и id на старте пустые
     images = []
     labels = []

     # Перебираем все картинки в датасете
     for image_path in image_paths:
         # Читаем картинку и сразу переводим в ч/б
         image_pil = Image.open(image_path).convert('L')
         # Переводим картинку в numpy-массив
         image = np.array(image_pil, 'uint8')
         # Получаем id пользователя из имени файла
         nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
         # Определяем лицо на картинке
         faces = faceCascade.detectMultiScale(image)
         # Если лицо найдено
         for (x, y, w, h) in faces:
             # Добавляем его к списку картинок
             images.append(image[y: y + h, x: x + w])
             # Добавляем id пользователя в список id
             labels.append(nbr)
             # Выводим текущую картинку на экран
             cv2.imshow("Adding faces to training set...", image[y: y + h, x: x + w])
             # Делаем паузу
             cv2.waitKey(100)
     # Возвращаем список картинок и id
     return images, labels

# Получаем список картинок и id
images, labels = get_images_and_labels(dataPath)
# Обучаем модель распознавания на наших картинках и учим сопоставлять её лица и id к ним
recognizer.train(images, np.array(labels))
# Сохраняем модель
recognizer.save(path + r'/trainer/trainer.yml')
# Удаляем из памяти все созданные окна
cv2.destroyAllWindows()