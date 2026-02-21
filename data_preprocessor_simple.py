import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class StickDataPreprocessor:
    def __init__(self, image_size=(640, 640)):
        """
        Инициализация препроцессора данных для палочек
        
        Args:
            image_size: размер изображений для модели
        """
        self.image_size = image_size
    
    def load_image(self, image_path):
        """
        Загрузка изображения
        
        Args:
            image_path: путь к изображению
            
        Returns:
            изображение в формате numpy array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def enhance_stick_contrast(self, image):
        """
        Улучшение контрастности для лучшей детекции палочек
        
        Args:
            image: входное изображение
            
        Returns:
            улучшенное изображение
        """
        # Конвертация в grayscale для анализа
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Адаптивная гистограммная эквализация
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Обратная конвертация в RGB
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # Смешивание с оригиналом
        result = cv2.addWeighted(image, 0.7, enhanced_rgb, 0.3, 0)
        
        return result
    
    def detect_edges(self, image):
        """
        Детекция границ для выделения палочек
        
        Args:
            image: входное изображение
            
        Returns:
            изображение с выделенными границами
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Размытие для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Детектор границ Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Морфологические операции для улучшения
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def preprocess_training_image(self, image_path, is_training=True):
        """
        Предобработка тренировочного изображения
        
        Args:
            image_path: путь к изображению
            is_training: флаг тренировки/валидации
            
        Returns:
            обработанное изображение
        """
        image = self.load_image(image_path)
        
        # Улучшение контрастности
        image = self.enhance_stick_contrast(image)
        
        # Изменение размера
        image = cv2.resize(image, self.image_size)
        
        return image
    
    def create_synthetic_sticks(self, output_dir, num_images=100):
        """
        Создание синтетических изображений палочек для тренировки
        
        Args:
            output_dir: директория для сохранения
            num_images: количество изображений
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        
        for i in tqdm(range(num_images), desc="Создание синтетических данных"):
            # Создание пустого изображения
            image = np.random.randint(200, 255, (640, 640, 3), dtype=np.uint8)
            
            # Добавление фонового шума
            noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)
            
            # Генерация случайных палочек
            num_sticks = np.random.randint(1, 5)
            stick_annotations = []
            
            for _ in range(num_sticks):
                # Случайные параметры палочки
                length = np.random.randint(50, 200)
                width = np.random.randint(2, 8)
                angle = np.random.uniform(0, 180)
                
                # Случайная позиция
                center_x = np.random.randint(length, 640 - length)
                center_y = np.random.randint(length, 640 - length)
                
                # Создание палочки
                stick_color = np.random.randint(50, 150, 3)
                
                # Рисование палочки с поворотом
                rect = ((center_x, center_y), (length, width), angle)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                cv2.drawContours(image, [box], 0, stick_color, -1)
                
                # Добавление текстуры
                cv2.drawContours(image, [box], 0, stick_color - 20, 1)
                
                # Конвертация в YOLO формат (нормализованные координаты)
                x_center = center_x / 640
                y_center = center_y / 640
                width_norm = length / 640
                height_norm = width / 640
                
                stick_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
            
            # Сохранение изображения
            image_path = os.path.join(output_dir, f"synthetic_{i:04d}.jpg")
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Сохранение аннотации
            label_path = os.path.join(output_dir, 'labels', f"synthetic_{i:04d}.txt")
            with open(label_path, 'w') as f:
                f.write('\n'.join(stick_annotations))
        
        print(f"Создано {num_images} синтетических изображений в {output_dir}")
    
    def analyze_dataset_statistics(self, images_dir, labels_dir):
        """
        Анализ статистики датасета
        
        Args:
            images_dir: директория с изображениями
            labels_dir: директория с разметкой
        """
        stats = {
            'total_images': 0,
            'total_sticks': 0,
            'stick_sizes': [],
            'aspect_ratios': []
        }
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for image_file in tqdm(image_files, desc="Анализ датасета"):
            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            if os.path.exists(label_path):
                stats['total_images'] += 1
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # YOLO формат: class x_center y_center width height
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            stats['total_sticks'] += 1
                            stats['stick_sizes'].append(width * height)
                            stats['aspect_ratios'].append(width / height if height > 0 else 0)
        
        # Вывод статистики
        print(f"Статистика датасета:")
        print(f"  - Всего изображений: {stats['total_images']}")
        print(f"  - Всего палочек: {stats['total_sticks']}")
        print(f"  - Среднее количество палочек на изображение: {stats['total_sticks'] / max(stats['total_images'], 1):.2f}")
        
        if stats['stick_sizes']:
            print(f"  - Средний размер палочки: {np.mean(stats['stick_sizes']):.4f}")
            print(f"  - Среднее соотношение сторон: {np.mean(stats['aspect_ratios']):.2f}")
        
        return stats
    
    def visualize_preprocessing_steps(self, image_path):
        """
        Визуализация этапов предобработки
        
        Args:
            image_path: путь к изображению
        """
        original = self.load_image(image_path)
        enhanced = self.enhance_stick_contrast(original)
        edges = self.detect_edges(enhanced)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original)
        axes[0].set_title('Оригинал')
        axes[0].axis('off')
        
        axes[1].imshow(enhanced)
        axes[1].set_title('Улучшенная контрастность')
        axes[1].axis('off')
        
        axes[2].imshow(edges, cmap='gray')
        axes[2].set_title('Детекция границ')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Пример использования
    preprocessor = StickDataPreprocessor()
    
    # Создание синтетических данных
    preprocessor.create_synthetic_sticks("synthetic_data", num_images=50)
    
    # Анализ созданного датасета
    stats = preprocessor.analyze_dataset_statistics("synthetic_data", "synthetic_data/labels")
    
    # Визуализация предобработки (если есть тестовое изображение)
    test_image = "synthetic_data/synthetic_0000.jpg"
    if os.path.exists(test_image):
        preprocessor.visualize_preprocessing_steps(test_image)
