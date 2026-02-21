import torch
import torch.nn as nn
from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class StickModelTrainer:
    def __init__(self, dataset_path, model_name='yolov8n.pt'):
        """
        Инициализация тренера модели для детекции палочек
        
        Args:
            dataset_path: путь к датасету
            model_name: имя базовой модели YOLO
        """
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.model = None
        self.training_history = []
        
    def prepare_dataset_config(self):
        """
        Подготовка конфигурационного файла датасета
        """
        config = {
            'path': os.path.abspath(self.dataset_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images' if os.path.exists(os.path.join(self.dataset_path, 'test/images')) else 'val/images',
            'nc': 1,  # количество классов
            'names': ['stick']  # имена классов
        }
        
        config_path = os.path.join(self.dataset_path, 'dataset_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_path
    
    def validate_dataset(self):
        """
        Валидация структуры датасета
        """
        required_dirs = [
            'train/images',
            'train/labels',
            'val/images',
            'val/labels'
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = os.path.join(self.dataset_path, dir_path)
            if not os.path.exists(full_path):
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            raise ValueError(f"Отсутствуют директории: {missing_dirs}")
        
        # Проверка соответствия изображений и разметки
        for split in ['train', 'val']:
            img_dir = os.path.join(self.dataset_path, f'{split}/images')
            label_dir = os.path.join(self.dataset_path, f'{split}/labels')
            
            img_files = {f.split('.')[0] for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))}
            label_files = {f.split('.')[0] for f in os.listdir(label_dir) if f.endswith('.txt')}
            
            missing_labels = img_files - label_files
            missing_images = label_files - img_files
            
            if missing_labels:
                print(f"Предупреждение: отсутствуют разметки для {len(missing_labels)} изображений в {split}")
            if missing_images:
                print(f"Предупреждение: отсутствуют изображения для {len(missing_images)} разметок в {split}")
        
        print("✅ Структура датасета корректна")
        return True
    
    def setup_model(self):
        """
        Настройка модели YOLO
        """
        self.model = YOLO(self.model_name)
        
        # Настройка гиперпараметров для детекции палочек
        self.model.model.nc = 1  # количество классов
        
        print(f"✅ Модель {self.model_name} загружена и настроена")
    
    def train(self, epochs=100, batch_size=16, img_size=640, device='auto'):
        """
        Обучение модели
        
        Args:
            epochs: количество эпох
            batch_size: размер батча
            img_size: размер изображений
            device: устройство для обучения
        """
        if self.model is None:
            self.setup_model()
        
        # Подготовка конфигурации датасета
        config_path = self.prepare_dataset_config()
        
        # Параметры обучения
        training_params = {
            'data': config_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': device,
            'project': 'stick_detection',
            'name': 'stick_model',
            'save_period': 10,
            'patience': 20,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
        
        print(f"🚀 Начало обучения модели...")
        print(f"   - Эпохи: {epochs}")
        print(f"   - Размер батча: {batch_size}")
        print(f"   - Размер изображений: {img_size}")
        
        # Обучение модели
        try:
            results = self.model.train(**training_params)
            
            # Сохранение истории обучения
            self.training_history = results.results_dict
            
            print("✅ Обучение завершено успешно!")
            return results
            
        except Exception as e:
            print(f"❌ Ошибка при обучении: {str(e)}")
            return None
    
    def evaluate_model(self, model_path=None):
        """
        Оценка качества модели
        
        Args:
            model_path: путь к обученной модели
        """
        if model_path:
            eval_model = YOLO(model_path)
        elif self.model:
            eval_model = self.model
        else:
            raise ValueError("Модель не загружена")
        
        # Подготовка конфигурации датасета
        config_path = self.prepare_dataset_config()
        
        print("📊 Оценка качества модели...")
        
        # Валидация модели
        metrics = eval_model.val(data=config_path)
        
        # Вывод метрик
        print(f"   - mAP50: {metrics.box.map50:.4f}")
        print(f"   - mAP50-95: {metrics.box.map:.4f}")
        print(f"   - Precision: {metrics.box.mp:.4f}")
        print(f"   - Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def export_model(self, format='onnx'):
        """
        Экспорт модели в различные форматы
        
        Args:
            format: формат экспорта (onnx, torchscript, coreml)
        """
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        print(f"📦 Экспорт модели в формат {format}...")
        
        # Экспорт модели
        exported_path = self.model.export(format=format)
        
        print(f"✅ Модель экспортирована: {exported_path}")
        return exported_path
    
    def plot_training_history(self):
        """
        Визуализация истории обучения
        """
        if not self.training_history:
            print("История обучения недоступна")
            return
        
        # Создание графиков
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        if 'train/box_loss' in self.training_history:
            epochs = range(1, len(self.training_history['train/box_loss']) + 1)
            axes[0, 0].plot(epochs, self.training_history['train/box_loss'], label='Потери на обучении')
            axes[0, 0].plot(epochs, self.training_history['val/box_loss'], label='Потери на валидации')
            axes[0, 0].set_title('Потери bounding box')
            axes[0, 0].legend()
        
        # mAP
        if 'metrics/mAP50' in self.training_history:
            epochs = range(1, len(self.training_history['metrics/mAP50']) + 1)
            axes[0, 1].plot(epochs, self.training_history['metrics/mAP50'], label='mAP50')
            axes[0, 1].plot(epochs, self.training_history['metrics/mAP50-95'], label='mAP50-95')
            axes[0, 1].set_title('Средняя точность')
            axes[0, 1].legend()
        
        # Precision и Recall
        if 'metrics/precision' in self.training_history:
            epochs = range(1, len(self.training_history['metrics/precision']) + 1)
            axes[1, 0].plot(epochs, self.training_history['metrics/precision'], label='Точность')
            axes[1, 0].plot(epochs, self.training_history['metrics/recall'], label='Полнота')
            axes[1, 0].set_title('Точность и Полнота')
            axes[1, 0].legend()
        
        # F1 Score
        if 'metrics/F1' in self.training_history:
            epochs = range(1, len(self.training_history['metrics/F1']) + 1)
            axes[1, 1].plot(epochs, self.training_history['metrics/F1'], label='F1 метрика')
            axes[1, 1].set_title('F1 метрика')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_training_report(self, output_path='training_report.json'):
        """
        Сохранение отчета об обучении
        
        Args:
            output_path: путь для сохранения отчета
        """
        report = {
            'model_name': self.model_name,
            'dataset_path': self.dataset_path,
            'training_history': self.training_history,
            'timestamp': str(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Отчет сохранен: {output_path}")

def main():
    """
    Основная функция для запуска обучения
    """
    print("🎯 Обучение модели детекции палочек")
    print("=" * 50)
    
    # Параметры обучения
    dataset_path = "synthetic_data"  # путь к датасету
    epochs = 50
    batch_size = 8
    
    # Создание тренера
    trainer = StickModelTrainer(dataset_path)
    
    try:
        # Валидация датасета
        trainer.validate_dataset()
        
        # Обучение модели
        results = trainer.train(epochs=epochs, batch_size=batch_size)
        
        if results:
            # Оценка модели
            metrics = trainer.evaluate_model()
            
            # Визуализация результатов
            trainer.plot_training_history()
            
            # Сохранение отчета
            trainer.save_training_report()
            
            # Экспорт модели
            trainer.export_model('onnx')
            
            print("🎉 Обучение завершено успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")

if __name__ == "__main__":
    main()
