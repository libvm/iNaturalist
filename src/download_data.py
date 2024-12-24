import pandas as pd
import os
import requests
from urllib.parse import urlparse
from pathlib import Path


def download_images(csv_file, target_dir):
    """
    Скачивает изображения из CSV файла и сохраняет их в директорию target_dir.
    Названия файлов сохраняются с оригинальными расширениями.
    """
    os.makedirs(target_dir, exist_ok=True)
    data = pd.read_csv(csv_file)

    for idx, row in data.iterrows():
        url, _ = row
        try:
            # Извлекаем оригинальное расширение из URL
            file_extension = Path(urlparse(url).path).suffix
            if (
                not file_extension
            ):  # Если расширение не указано, используем .jpg по умолчанию
                file_extension = ".jpg"

            image_name = f"{target_dir}/{idx}{file_extension}"

            if not os.path.exists(image_name):
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(image_name, "wb") as file:
                        file.write(response.content)
                else:
                    print(f"Ошибка загрузки {url}: статус {response.status_code}")
        except Exception as e:
            print(f"Ошибка обработки URL {url}: {e}")


def download_data():
    categories = [
        "foxes",
        "raccoons",
        "weasels",
        "skunks",
        "hyenas",
        "mongooses",
        "cats",
        "wolves",
    ]
    for category in categories:
        download_images(f"data/{category}.csv", f"images/{category}")


if __name__ == "__main__":
    download_data()
