@echo off
echo 🐍 Установка Python и зависимостей
echo ===================================

echo.
echo 🔍 Проверка установки Python...
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Python уже установлен
    python --version
) else (
    echo ❌ Python не найден
    echo 📥 Пожалуйста, установите Python с официального сайта:
    echo    https://www.python.org/downloads/
    echo    Выберите версию 3.8 или новее
    echo    При установке отметьте "Add Python to PATH"
    pause
    exit /b 1
)

echo.
echo 📦 Установка зависимостей...
python -m pip install --upgrade pip

echo.
echo 🔧 Установка OpenCV...
python -m pip install opencv-python

echo.
echo 🔧 Установка NumPy...
python -m pip install numpy

echo.
echo 🔧 Установка Pillow...
python -m pip install Pillow

echo.
echo 🔧 Установка дополнительных библиотек...
python -m pip install matplotlib
python -m pip install scikit-learn
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics
python -m pip install tqdm
python -m pip install albumentations

echo.
echo ✅ Установка завершена!
echo.
echo 🚀 Запуск демонстрации:
python simple_demo.py
echo.
echo 🌐 Для веб-интерфейса (если установлен streamlit):
streamlit run app.py
echo.

pause
