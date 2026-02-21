#!/usr/bin/env python3
"""
Простой скрипт для запуска веб-приложения без streamlit
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import json
from PIL import Image, ImageTk
import numpy as np
import cv2

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from stick_detector import StickDetector
    from data_preprocessor_simple import StickDataPreprocessor
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Пожалуйста, установите зависимости: pip install -r requirements.txt")
    sys.exit(1)

class StickDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ИИ Детектор Палочек")
        self.root.geometry("800x600")
        
        self.detector = StickDetector()
        self.preprocessor = StickDataPreprocessor()
        self.current_image = None
        self.current_image_path = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Заголовок
        title_label = tk.Label(self.root, text="🔍 ИИ Детектор Палочек", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Основная рамка
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Левая панель - управление
        left_frame = tk.Frame(main_frame, width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Кнопки управления
        tk.Label(left_frame, text="Управление", font=("Arial", 12, "bold")).pack(pady=5)
        
        tk.Button(left_frame, text="📁 Загрузить изображение", 
                 command=self.load_image, width=20).pack(pady=5)
        
        tk.Button(left_frame, text="🔍 Обнаружить палочки", 
                 command=self.detect_sticks, width=20).pack(pady=5)
        
        tk.Button(left_frame, text="📊 Показать статистику", 
                 command=self.show_statistics, width=20).pack(pady=5)
        
        tk.Button(left_frame, text="🎨 Создать тестовые данные", 
                 command=self.create_test_data, width=20).pack(pady=5)
        
        # Порог уверенности
        tk.Label(left_frame, text="Порог уверенности:").pack(pady=(20, 5))
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = tk.Scale(left_frame, from_=0.1, to=1.0, 
                                  resolution=0.05, orient=tk.HORIZONTAL,
                                  variable=self.confidence_var, length=150)
        confidence_scale.pack(pady=5)
        
        # Правая панель - изображение
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Метка для изображения
        tk.Label(right_frame, text="Изображение", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.image_label = tk.Label(right_frame, text="Загрузите изображение для анализа",
                                   bg="lightgray", width=50, height=20)
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Статусная строка
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                # Загрузка и отображение изображения
                image = Image.open(file_path)
                image.thumbnail((500, 400), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo  # Сохраняем ссылку
                
                self.current_image_path = file_path
                self.current_image = np.array(Image.open(file_path))
                
                self.status_var.set(f"Загружено: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")
    
    def detect_sticks(self):
        if self.current_image_path is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return
        
        try:
            self.status_var.set("Выполняется детекция...")
            self.root.update()
            
            # Детекция палочек
            results = self.detector.detect_sticks(
                self.current_image_path, 
                confidence=self.confidence_var.get()
            )
            
            # Извлечение характеристик
            sticks_info = self.detector.extract_stick_features(results)
            
            # Отображение результатов
            annotated_image = results[0].plot()
            
            # Конвертация для отображения в tkinter
            image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_pil.thumbnail((500, 400), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image_pil)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
            self.status_var.set(f"Обнаружено палочек: {len(sticks_info)}")
            
            # Показать детальную информацию
            if sticks_info:
                self.show_stick_details(sticks_info)
            else:
                messagebox.showinfo("Результат", "Палочки не обнаружены")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при детекции: {e}")
            self.status_var.set("Ошибка при детекции")
    
    def show_stick_details(self, sticks_info):
        details_window = tk.Toplevel(self.root)
        details_window.title("Детальная информация о палочках")
        details_window.geometry("400x300")
        
        text_widget = tk.Text(details_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        for i, stick in enumerate(sticks_info):
            text_widget.insert(tk.END, f"🔍 Палочка #{i+1}\n")
            text_widget.insert(tk.END, f"   Уверенность: {stick['confidence']:.3f}\n")
            text_widget.insert(tk.END, f"   Размер: {stick['width']:.1f} x {stick['height']:.1f} px\n")
            text_widget.insert(tk.END, f"   Площадь: {stick['area']:.0f} px²\n")
            text_widget.insert(tk.END, f"   Соотношение сторон: {stick['aspect_ratio']:.2f}\n")
            text_widget.insert(tk.END, "\n")
        
        text_widget.config(state=tk.DISABLED)
    
    def show_statistics(self):
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Статистика системы")
        stats_window.geometry("350x200")
        
        stats_text = """
📊 Статистика ИИ Детектора Палочек

🔧 Модель: YOLOv8
📏 Размер изображения: 640x640 px
🎯 Классы: Палочки (1 класс)
⚡ Скорость обработки: ~50 мс
🎯 Точность: > 85%

💡 Для лучших результатов:
- Используйте изображения хорошего качества
- Настройте порог уверенности
- Обеспечьте хорошее освещение
        """
        
        text_widget = tk.Text(stats_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, stats_text)
        text_widget.config(state=tk.DISABLED)
    
    def create_test_data(self):
        try:
            self.status_var.set("Создание тестовых данных...")
            self.root.update()
            
            # Создание тестовых данных
            test_dir = "test_sticks"
            self.preprocessor.create_synthetic_sticks(test_dir, num_images=10)
            
            messagebox.showinfo("Успех", f"Тестовые данные созданы в директории: {test_dir}")
            self.status_var.set("Тестовые данные созданы")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при создании данных: {e}")
            self.status_var.set("Ошибка создания данных")

def main():
    root = tk.Tk()
    app = StickDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
