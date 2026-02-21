import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import os
import tempfile
from stick_detector import StickDetector
from data_preprocessor_simple import StickDataPreprocessor
import matplotlib.pyplot as plt
import json

# Настройка страницы
st.set_page_config(
    page_title="ИИ Детектор Палочек",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("🔍 ИИ Детектор Палочек")
st.markdown("---")
st.markdown("### Интеллектуальная система для обнаружения и анализа палочек на изображениях")

# Инициализация детектора в session_state
if 'detector' not in st.session_state:
    st.session_state.detector = StickDetector()
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = StickDataPreprocessor()

# Боковая панель
st.sidebar.header("Настройки детекции")

# Порог уверенности
confidence_threshold = st.sidebar.slider(
    "Порог уверенности",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Режим работы
mode = st.sidebar.selectbox(
    "Режим работы",
    ["Детекция на изображении", "Анализ датасета", "Создание синтетических данных"]
)

# Основной контент
if mode == "Детекция на изображении":
    st.header("📸 Детекция палочек на изображении")
    
    # Загрузка изображения
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Загрузка изображения")
        uploaded_file = st.file_uploader(
            "Выберите изображение",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file is not None:
            # Сохранение временного файла
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Отображение оригинального изображения
            image = Image.open(tmp_path)
            st.image(image, caption="Загруженное изображение", use_column_width=True)
            
            # Кнопка детекции
            if st.button("🔍 Обнаружить палочки", type="primary"):
                with st.spinner("Выполняется детекция..."):
                    try:
                        # Детекция палочек
                        results = st.session_state.detector.detect_sticks(
                            tmp_path, 
                            confidence=confidence_threshold
                        )
                        
                        # Извлечение характеристик
                        sticks_info = st.session_state.detector.extract_stick_features(results)
                        
                        # Отображение результатов во второй колонке
                        with col2:
                            st.subheader("Результаты детекции")
                            
                            # Визуализация
                            annotated_image = results[0].plot()
                            st.image(annotated_image, caption="Обнаруженные палочки", use_column_width=True)
                            
                            # Статистика
                            st.subheader("📊 Статистика обнаружения")
                            
                            col_stats1, col_stats2, col_stats3 = st.columns(3)
                            
                            with col_stats1:
                                st.metric(
                                    "Обнаружено палочек",
                                    len(sticks_info)
                                )
                            
                            with col_stats2:
                                if sticks_info:
                                    avg_confidence = np.mean([stick['confidence'] for stick in sticks_info])
                                    st.metric(
                                        "Средняя уверенность",
                                        f"{avg_confidence:.2f}"
                                    )
                                else:
                                    st.metric("Средняя уверенность", "0.00")
                            
                            with col_stats3:
                                if sticks_info:
                                    avg_area = np.mean([stick['area'] for stick in sticks_info])
                                    st.metric(
                                        "Средняя площадь",
                                        f"{avg_area:.0f} px²"
                                    )
                                else:
                                    st.metric("Средняя площадь", "0 px²")
                            
                            # Детальная информация о каждой палочке
                            if sticks_info:
                                st.subheader("📋 Детальная информация")
                                
                                for i, stick in enumerate(sticks_info):
                                    with st.expander(f"Палочка #{i+1}"):
                                        col_info1, col_info2 = st.columns(2)
                                        
                                        with col_info1:
                                            st.write(f"**Уверенность:** {stick['confidence']:.3f}")
                                            st.write(f"**Ширина:** {stick['width']:.1f} px")
                                            st.write(f"**Высота:** {stick['height']:.1f} px")
                                        
                                        with col_info2:
                                            st.write(f"**Соотношение сторон:** {stick['aspect_ratio']:.2f}")
                                            st.write(f"**Площадь:** {stick['area']:.0f} px²")
                                            st.write(f"**Координаты:** ({stick['bbox'][0]:.0f}, {stick['bbox'][1]:.0f})")
                            else:
                                st.warning("Палочки не обнаружены. Попробуйте снизить порог уверенности.")
                        
                        # Очистка временного файла
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"Ошибка при обработке изображения: {str(e)}")
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

elif mode == "Анализ датасета":
    st.header("📊 Анализ датасета")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Загрузка датасета")
        
        images_dir = st.text_input("Директория с изображениями:", "dataset/images")
        labels_dir = st.text_input("Директория с разметкой:", "dataset/labels")
        
        if st.button("📈 Проанализировать датасет"):
            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                with st.spinner("Анализ датасета..."):
                    try:
                        stats = st.session_state.preprocessor.analyze_dataset_statistics(
                            images_dir, labels_dir
                        )
                        
                        with col2:
                            st.subheader("Результаты анализа")
                            
                            # Визуализация статистики
                            if stats['stick_sizes']:
                                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                                
                                # Распределение размеров
                                axes[0].hist(stats['stick_sizes'], bins=20, alpha=0.7)
                                axes[0].set_title('Распределение размеров палочек')
                                axes[0].set_xlabel('Площадь (нормализованная)')
                                axes[0].set_ylabel('Количество')
                                
                                # Распределение соотношений сторон
                                axes[1].hist(stats['aspect_ratios'], bins=20, alpha=0.7, color='orange')
                                axes[1].set_title('Распределение соотношений сторон')
                                axes[1].set_xlabel('Ширина / Высота')
                                axes[1].set_ylabel('Количество')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                        
                        # Сохранение статистики
                        st.success("Анализ завершен!")
                        
                    except Exception as e:
                        st.error(f"Ошибка при анализе датасета: {str(e)}")
            else:
                st.error("Указанные директории не существуют")

elif mode == "Создание синтетических данных":
    st.header("🎨 Создание синтетических данных")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Параметры генерации")
        
        num_images = st.number_input(
            "Количество изображений",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
        
        output_dir = st.text_input(
            "Директория для сохранения:",
            "synthetic_sticks"
        )
        
        if st.button("🎲 Создать синтетические данные"):
            with st.spinner("Создание синтетических данных..."):
                try:
                    st.session_state.preprocessor.create_synthetic_sticks(
                        output_dir, num_images
                    )
                    
                    with col2:
                        st.subheader("Результаты генерации")
                        st.success(f"Создано {num_images} синтетических изображений!")
                        
                        # Показ примера
                        example_path = os.path.join(output_dir, "synthetic_0000.jpg")
                        if os.path.exists(example_path):
                            example_image = Image.open(example_path)
                            st.image(example_image, caption="Пример синтетического изображения", use_column_width=True)
                
                except Exception as e:
                    st.error(f"Ошибка при создании синтетических данных: {str(e)}")

# Информационная панель
st.markdown("---")
st.markdown("### ℹ️ Информация о системе")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.info("**Модель:** YOLOv8 с дообучением")

with col_info2:
    st.info("**Размер изображения:** 640x640 px")

with col_info3:
    st.info("**Классы:** Палочки (1 класс)")

# Инструкции
with st.expander("📖 Инструкции по использованию"):
    st.markdown("""
    ### Детекция на изображении:
    1. Загрузите изображение с палочками
    2. Настройте порог уверенности
    3. Нажмите "Обнаружить палочки"
    4. Изучите результаты и статистику
    
    ### Анализ датасета:
    1. Укажите пути к директориям с изображениями и разметкой
    2. Нажмите "Проанализировать датасет"
    3. Изучите статистику и распределения
    
    ### Создание синтетических данных:
    1. Укажите количество изображений
    2. Выберите директорию для сохранения
    3. Нажмите "Создать синтетические данные"
    4. Используйте созданные данные для тренировки модели
    """)

# Футер
st.markdown("---")
st.markdown("**Создано для AnaMed Forum** | ИИ система детекции палочек")
