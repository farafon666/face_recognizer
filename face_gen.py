# Библиотека машинного зрения
import cv2
# Библиотека для вызова системных функций
import os

# Получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))
# Указываем, что мы будем искать лица по примитивам Хаара
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# Счётчик изображений
i = 0
# Расстояния от распознанного лица до рамки
offset = 50
# Запрашиваем номер пользователя
name = input('Введите номер пользователя: ')
# Получаем доступ к камере
video = cv2.VideoCapture(0)

# Запускаем цикл
while True:
    # Берём видеопоток
    ret, im = video.read()
    # Переводим всё в ч/б для простоты
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Настраиваем параметры распознавания и получаем лицо с камеры
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
    # Обрабатываем лица
    for(x,y,w,h) in faces:
        # Увеличиваем счётчик кадров
        i = i+1
        # Записываем файл на диск
        cv2.imwrite("dataSet/face-" + name + '.' + str(i) + ".jpg", gray[y-offset:y+h+offset, x-offset:x+w+offset])
        # Формируем размеры окна для вывода лица
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        # Показываем очередной кадр, который мы запомнили
        cv2.imshow('im', im[y-offset:y+h+offset, x-offset:x+w+offset])
        # Делаем паузу
        cv2.waitKey(100)
    # Если у нас хватает кадров
    if i > 30:
        # Освобождаем камеру
        video.release()
        # Удалаяем все созданные окна
        cv2.destroyAllWindows()
        # Останавливаем цикл
        break