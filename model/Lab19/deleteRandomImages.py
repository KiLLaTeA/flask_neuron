import os
import random

# Укажите путь к папке с фотографиями
folder_path = 'cats'

# Получаем список всех файлов в папке
files = os.listdir(folder_path)

# Фильтруем только файлы с расширением изображений (например, .jpg, .png)
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
image_files = [f for f in files if f.lower().endswith(image_extensions)]

# Если в папке есть изображения
if image_files:
    # Определяем количество файлов для удаления (случайное число)
    num_to_delete = random.randint(1, len(image_files))  # Удаляем от 1 до всех файлов

    # Выбираем случайные файлы для удаления
    files_to_delete = random.sample(image_files, num_to_delete)

    # Удаляем выбранные файлы
    for file in files_to_delete:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
        print(f"Удален файл: {file}")
else:
    print("В папке нет изображений.")