import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

class StickDetector:
    def __init__(self, model_path=None):
        """
        Инициализация детектора палочек
        
        Args:
            model_path: путь к предобученной модели YOLO
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Используем предобученную YOLOv8n для начала
            self.model = YOLO('yolov8n.pt')
            self.fine_tune_for_sticks()
    
    def fine_tune_for_sticks(self):
        """
        Дообучение модели для детекции палочек
        """
        print("Модель готова для дообучения на данных палочек")
        # Здесь будет логика дообучения при наличии датасета
    
    def preprocess_image(self, image_path):
        """
        Предобработка изображения
        
        Args:
            image_path: путь к изображению
            
        Returns:
            обработанное изображение
        """
        # Загрузка изображения
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        # Конвертация в RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Изменение размера
        image = cv2.resize(image, (640, 640))
        
        return image
    
    def detect_sticks(self, image_path, confidence=0.5):
        """
        Детекция палочек на изображении
        
        Args:
            image_path: путь к изображению или numpy array
            confidence: порог уверенности
            
        Returns:
            результаты детекции
        """
        # Предобработка
        processed_image = self.preprocess_image(image_path)
        
        # Детекция
        results = self.model(processed_image, conf=confidence)
        
        return results
    
    def visualize_results(self, image_path, results, save_path=None):
        """
        Визуализация результатов детекции
        
        Args:
            image_path: путь к исходному изображению
            results: результаты детекции
            save_path: путь для сохранения результата
        """
        # Загрузка оригинального изображения
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # Отрисовка результатов
        annotated_image = results[0].plot()
        
        # Отображение
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.title('Результаты детекции палочек')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
        
        return annotated_image
    
    def extract_stick_features(self, results):
        """
        Извлечение характеристик обнаруженных палочек
        
        Args:
            results: результаты детекции
            
        Returns:
            список характеристик палочек
        """
        sticks_info = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Координаты bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Уверенность
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Класс (для палочек будем использовать специальный класс)
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Вычисление характеристик
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = width / height if height > 0 else 0
                    
                    stick_info = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'width': width,
                        'height': height,
                        'aspect_ratio': aspect_ratio,
                        'area': width * height
                    }
                    
                    sticks_info.append(stick_info)
        
        return sticks_info
    
    def create_training_dataset(self, images_dir, labels_dir, output_dir):
        """
        Создание тренировочного датасета
        
        Args:
            images_dir: директория с изображениями
            labels_dir: директория с разметкой
            output_dir: директория для сохранения датасета
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Структура YOLO датасета
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        
        os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
        
        # Создание YAML файла для YOLO
        yaml_content = f"""
path: {output_dir}
train: train/images
val: val/images

nc: 1  # количество классов (только палочки)
names: ['палочка']  # имена классов
"""
        
        with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
            f.write(yaml_content)
        
        print(f"Структура датасета создана в {output_dir}")

if __name__ == "__main__":
    # Пример использования
    detector = StickDetector()
    
    # Проверка детекции на тестовом изображении
    test_image_path = "test_stick_image.jpg"  # замените на реальный путь
    
    if os.path.exists(test_image_path):
        results = detector.detect_sticks(test_image_path)
        sticks_info = detector.extract_stick_features(results)
        
        print(f"Обнаружено палочек: {len(sticks_info)}")
        for i, stick in enumerate(sticks_info):
            print(f"Палочка {i+1}:")
            print(f"  - Уверенность: {stick['confidence']:.2f}")
            print(f"  - Размер: {stick['width']:.1f} x {stick['height']:.1f}")
            print(f"  - Соотношение сторон: {stick['aspect_ratio']:.2f}")
        
        # Визуализация
        detector.visualize_results(test_image_path, results)
    else:
        print("Тестовое изображение не найдено. Создайте детектор и загрузите изображение.")
