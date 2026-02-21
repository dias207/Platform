#!/usr/bin/env python3
"""
Простая демонстрация детектора палочек без GUI
"""

import os
import sys

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🔍 ИИ Детектор Палочек - Демонстрация")
    print("=" * 50)
    
    # Проверка базовых модулей
    try:
        import cv2
        print("✅ OpenCV доступен")
    except ImportError:
        print("❌ OpenCV не установлен")
        print("Установите: pip install opencv-python")
        return
    
    try:
        import numpy as np
        print("✅ NumPy доступен")
    except ImportError:
        print("❌ NumPy не установлен")
        print("Установите: pip install numpy")
        return
    
    try:
        from PIL import Image
        print("✅ PIL доступен")
    except ImportError:
        print("❌ PIL не установлен")
        print("Установите: pip install Pillow")
        return
    
    print("\n🚀 Создание тестового изображения...")
    
    # Создание тестового изображения с палочками
    import numpy as np
    
    # Создание пустого изображения
    image = np.random.randint(200, 255, (640, 640, 3), dtype=np.uint8)
    
    # Добавление палочек
    import cv2
    
    # Палочка 1 - горизонтальная
    cv2.rectangle(image, (100, 200), (300, 210), (50, 50, 50), -1)
    
    # Палочка 2 - вертикальная
    cv2.rectangle(image, (400, 100), (410, 300), (80, 80, 80), -1)
    
    # Палочка 3 - диагональная
    pts = np.array([[200, 400], [250, 450], [240, 460], [190, 410]], np.int32)
    cv2.fillPoly(image, [pts], (60, 60, 60))
    
    # Сохранение тестового изображения
    cv2.imwrite("test_sticks.jpg", image)
    print("✅ Тестовое изображение создано: test_sticks.jpg")
    
    print("\n📊 Анализ изображения...")
    
    # Конвертация в grayscale для анализа
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применение порогового значения
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"🔍 Обнаружено объектов: {len(contours)}")
    
    # Анализ контуров
    stick_count = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 100:  # Фильтрация маленьких объектов
            stick_count += 1
            
            # Получение bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            print(f"\n📏 Палочка #{stick_count}:")
            print(f"   - Позиция: ({x}, {y})")
            print(f"   - Размер: {w} x {h} px")
            print(f"   - Площадь: {area:.0f} px²")
            print(f"   - Соотношение сторон: {aspect_ratio:.2f}")
            
            # Рисование bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Сохранение результата
    cv2.imwrite("test_sticks_result.jpg", image)
    print(f"\n✅ Результат сохранен: test_sticks_result.jpg")
    
    print(f"\n📈 Статистика:")
    print(f"   - Всего палочек обнаружено: {stick_count}")
    print(f"   - Обработано контуров: {len(contours)}")
    
    # Создание простого отчета
    report = f"""
ИИ Детектор Палочек - Отчет
========================

Дата: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Результаты анализа:
- Обнаружено палочек: {stick_count}
- Исходное изображение: test_sticks.jpg
- Результат: test_sticks_result.jpg

Техническая информация:
- Метод: OpenCV контурный анализ
- Порог детекции: 150
- Минимальная площадь: 100 px²

Система готова к использованию!
"""
    
    with open("detection_report.txt", "w", encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Отчет сохранен: detection_report.txt")
    
    print("\n🎉 Демонстрация завершена успешно!")
    print("\n📁 Созданные файлы:")
    print("   - test_sticks.jpg (исходное изображение)")
    print("   - test_sticks_result.jpg (с разметкой)")
    print("   - detection_report.txt (отчет)")
    
    print("\n💡 Для просмотра результатов откройте изображения в любом просмотрщике.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("\n💡 Установите необходимые зависимости:")
        print("   pip install opencv-python numpy")
        input("\nНажмите Enter для выхода...")
