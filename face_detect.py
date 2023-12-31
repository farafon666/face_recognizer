# Библиотека компьютерного зрения
import cv2
# Библиотека для вызова системных функций
import os

# Получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))
# Создаём новый распознаватель лиц
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Добавляем в него модель, которую мы обучили
recognizer.read(path + r'/trainer/trainer.yml')
# Указываем, что мы будем искать лица по примитивам Хаара
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Получаем доступ к камере
cam = cv2.VideoCapture(0)
# Настраиваем шрифт для вывода подписей
font = cv2.FONT_HERSHEY_SIMPLEX

# Запускаем цикл
while True:
    # Получаем видеопоток
    ret, im = cam.read()
    # Переводим его в ч/б
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Определяем лица на видео
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    # Перебираем все найденные лица
    for(x,y,w,h) in faces:
        # Получаем id пользователя
        nbr_predicted, coord = recognizer.predict(gray[y:y+h, x:x+w])
        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        # Если мы знаем id пользователя
        if(nbr_predicted == 1):
             # Подставляем вместо него имя человека
             nbr_predicted = 'Vladislav Ermakov'
        # Добавляем текст к рамке
        cv2.putText(im, str(nbr_predicted), (x,y+h),font, 1.1, (0,255,0))
        # Выводим окно с изображением с камеры
        cv2.imshow('Face recognition.', im)
        # Делаем паузу
        cv2.waitKey(10)