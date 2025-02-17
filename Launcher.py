import streamlit as st  
from PIL import Image  
from ultralytics import YOLO  # Убедитесь, что библиотека установлена  

# Заголовок приложения  
st.title("Загрузка и просмотр изображений")  

# Установка конфигурации загрузки изображений  
uploaded_files = st.file_uploader("Выберите изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True)  

# Переменные для хранения выбранного изображения и обработанных изображений в сессии  
if 'selected_image' not in st.session_state:  
    st.session_state.selected_image = None  

if 'processed_images' not in st.session_state:  
    st.session_state.processed_images = []  # Список для хранения обработанных изображений  

# Проверка, загружены ли изображения  
if uploaded_files:  
    # Отображение загруженных изображений  
    st.subheader("Список загруженных изображений:")  

    # Создаем контейнер для горизонтального списка  
    cols = st.columns(len(uploaded_files))  

    # Обходим загруженные файлы  
    for idx, uploaded_file in enumerate(uploaded_files):  
        img = Image.open(uploaded_file)  
        with cols[idx]:  
            # Отображение изображений  
            st.image(img, caption=uploaded_file.name, use_container_width=True)  

            # Кнопка для выбора изображения  
            if st.button(f"Обработать", key=uploaded_file.name):  
                st.session_state.selected_image = img  # Сохраняем выбранное изображение   
                st.session_state.processed_images = []  # Сбрасываем предыдущие обработанные изображения  

    # Отображаем выбранное изображение, если оно есть  
    if st.session_state.selected_image:  
         
        
        col1, col2 = st.columns(2)  
    
    # Отображаем изображения в соответствующих колонках  
        model_path = "runs/segment/XL_Original_300_epochs/weights/best.pt"  # Примечание: Замените на корректный путь  
        model = YOLO(model_path)  # Загружаем модель  
        
        # Если обработанные изображения еще не загружены, генерируем их  
        if not st.session_state.processed_images:  
            confidence_levels = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]  # Уровни уверенности  
            
            for conf in confidence_levels:  
                result = model.predict(st.session_state.selected_image, show_boxes=False, imgsz=832, iou=1, conf=conf, max_det=1000)  
                st.session_state.processed_images.append(result)  

        # Слайдер для выбора уровня уверенности  
        slider_value = st.slider("Чем выше значение, тем нейросеть более уверена в том, что это болото (иногда странно работает)",   
                                1, len(st.session_state.processed_images), 1)  
        
        selected_image = st.session_state.processed_images[slider_value - 1]
 
        with col1:
            st.subheader("Выбранное изображение:")   
            st.image(st.session_state.selected_image, caption="Выбранное изображение", use_container_width=True)  
        with col2:  
            st.subheader("Обработанное изображение:")   
            
            model_path = "runs/segment/XL_Original_300_epochs/weights/best.pt"  # Примечание: Замените на корректный путь  
            model = YOLO(model_path)  # Загружаем модель  
            
            # Если обработанные изображения еще не загружены, генерируем их  
            if not st.session_state.processed_images:  
                confidence_levels = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]  # Уровни уверенности  
                
                for conf in confidence_levels:  
                    result = model.predict(st.session_state.selected_image, show_boxes=False, imgsz=832, iou=1, conf=conf, max_det=1000)  
                    st.session_state.processed_images.append(result)  

            # Слайдер для выбора уровня уверенности  
            
            
            selected_image = st.session_state.processed_images[slider_value - 1]  
            
            # Отображаем первое предсказанное изображение  
            st.image(selected_image[0].plot(boxes=False), caption="Обработанное изображение", use_container_width=True)  # Обратите внимание на метод plot()  

else:  
    st.warning("Пожалуйста, загрузите хотя бы одно изображение.")