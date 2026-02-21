"""
Тестовый скрипт для проверки функциональности приложения
"""

import os
import sys
import traceback

def test_imports():
    """Проверка импортов основных модулей"""
    print("🔍 Проверка импортов...")
    
    try:
        import torch
        print("✅ PyTorch импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта PyTorch: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта NumPy: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта PIL: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта Matplotlib: {e}")
        return False
    
    return True

def test_local_modules():
    """Проверка локальных модулей"""
    print("\n🔍 Проверка локальных модулей...")
    
    try:
        from stick_detector import StickDetector
        print("✅ StickDetector импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта StickDetector: {e}")
        traceback.print_exc()
        return False
    
    try:
        from data_preprocessor_simple import StickDataPreprocessor
        print("✅ StickDataPreprocessor импортирован")
    except ImportError as e:
        print(f"❌ Ошибка импорта StickDataPreprocessor: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_detector_creation():
    """Проверка создания детектора"""
    print("\n🔍 Проверка создания детектора...")
    
    try:
        from stick_detector import StickDetector
        
        # Создание детектора (без загрузки модели)
        detector = StickDetector()
        print("✅ Детектор создан успешно")
        
        # Проверка методов
        assert hasattr(detector, 'detect_sticks'), "Метод detect_sticks отсутствует"
        assert hasattr(detector, 'preprocess_image'), "Метод preprocess_image отсутствует"
        assert hasattr(detector, 'extract_stick_features'), "Метод extract_stick_features отсутствует"
        print("✅ Все методы детектора доступны")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при создании детектора: {e}")
        traceback.print_exc()
        return False

def test_preprocessor():
    """Проверка препроцессора"""
    print("\n🔍 Проверка препроцессора...")
    
    try:
        from data_preprocessor import StickDataPreprocessor
        
        preprocessor = StickDataPreprocessor()
        print("✅ Препроцессор создан успешно")
        
        # Проверка методов
        assert hasattr(preprocessor, 'enhance_stick_contrast'), "Метод enhance_stick_contrast отсутствует"
        assert hasattr(preprocessor, 'detect_edges'), "Метод detect_edges отсутствует"
        assert hasattr(preprocessor, 'create_synthetic_sticks'), "Метод create_synthetic_sticks отсутствует"
        print("✅ Все методы препроцессора доступны")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при создании препроцессора: {e}")
        traceback.print_exc()
        return False

def test_synthetic_data_creation():
    """Проверка создания синтетических данных"""
    print("\n🔍 Проверка создания синтетических данных...")
    
    try:
        from data_preprocessor import StickDataPreprocessor
        
        preprocessor = StickDataPreprocessor()
        
        # Создание тестовой директории
        test_dir = "test_synthetic"
        
        # Создание небольшого набора синтетических данных
        preprocessor.create_synthetic_sticks(test_dir, num_images=5)
        
        # Проверка созданных файлов
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
            labels = [f for f in os.listdir(os.path.join(test_dir, 'labels')) if f.endswith('.txt')]
            
            if len(images) == 5 and len(labels) == 5:
                print("✅ Синтетические данные созданы успешно")
                
                # Очистка тестовых данных
                import shutil
                shutil.rmtree(test_dir)
                print("✅ Тестовые данные удалены")
                
                return True
            else:
                print(f"❌ Некорректное количество файлов: изображений {len(images)}, разметок {len(labels)}")
                return False
        else:
            print("❌ Директория с синтетическими данными не создана")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при создании синтетических данных: {e}")
        traceback.print_exc()
        return False

def test_streamlit_app():
    """Проверка импорта Streamlit приложения"""
    print("\n🔍 Проверка Streamlit приложения...")
    
    try:
        import streamlit as st
        print("✅ Streamlit импортирован")
        
        # Проверка синтаксиса файла app.py
        with open('app.py', 'r', encoding='utf-8') as f:
            app_content = f.read()
        
        # Компиляция для проверки синтаксиса
        compile(app_content, 'app.py', 'exec')
        print("✅ Синтаксис app.py корректен")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Синтаксическая ошибка в app.py: {e}")
        return False
    except ImportError as e:
        print(f"❌ Ошибка импорта Streamlit: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка при проверке app.py: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("🚀 Запуск тестирования ИИ Детектора Палочек")
    print("=" * 50)
    
    tests = [
        ("Импорт основных библиотек", test_imports),
        ("Импорт локальных модулей", test_local_modules),
        ("Создание детектора", test_detector_creation),
        ("Создание препроцессора", test_preprocessor),
        ("Создание синтетических данных", test_synthetic_data_creation),
        ("Проверка Streamlit приложения", test_streamlit_app)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Тест: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - ПРОЙДЕН")
            else:
                print(f"❌ {test_name} - НЕ ПРОЙДЕН")
        except Exception as e:
            print(f"❌ {test_name} - ОШИБКА: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Результаты тестирования: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 Все тесты пройдены! Приложение готово к использованию.")
        print("\n🚀 Для запуска веб-интерфейса выполните:")
        print("   streamlit run app.py")
    else:
        print("⚠️ Некоторые тесты не пройдены. Проверьте ошибки выше.")
        print("\n💡 Установите недостающие зависимости:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
