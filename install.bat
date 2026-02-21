@echo off
echo 🚀 Установка AI Детектора Палочек
echo ================================

echo.
echo 📦 Установка зависимостей...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python
pip install numpy
pip install matplotlib
pip install Pillow
pip install scikit-learn
pip install streamlit
pip install ultralytics
pip install tqdm
pip install albumentations

echo.
echo ✅ Установка завершена!
echo.
echo 🚀 Для запуска приложения выполните:
echo    streamlit run app.py
echo.
echo 🧪 Для тестирования выполните:
echo    python test_app.py
echo.

pause
