import streamlit as st  
from PIL import Image  
from ultralytics import YOLO  # Убедитесь, что библиотека установлена  
import cv2 
from modules.mask  import *


# Заголовок приложения  
st.title("Загрузка и просмотр изображений")  

# Установка конфигурации загрузки изображений  
uploaded_files = st.file_uploader("Выберите изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True)  

# Переменные для хранения выбранного изображения и обработанных изображений в сессии  
if 'selected_image' not in st.session_state:  
    st.session_state.selected_image = None  

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
            if st.button(f"Обработать {uploaded_file.name}", key=f"process_{uploaded_file.name}"):  
                st.session_state.selected_image = img  # Сохраняем выбранное изображение   
                st.session_state.processed_images = []  # Сбрасываем предыдущие обработанные изображения  

    # Отображаем выбранное изображение, если оно есть  
    if st.session_state.selected_image:  
        st.subheader("Выбранное изображение:")  
        
        st.image(st.session_state.selected_image, caption="Выбранное изображение", use_container_width=True)  
        
        st.subheader("Обработанное изображение:")   
 
        
        model_path = "./best.pt"  # Примечание: Замените на корректный путь  
        model = YOLO(model_path)  # Загружаем модель  

        # Если обработанные изображения еще не загружены, генерируем их    
        confidence_levels = [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006]  # Уровни уверенности  
            

        # Если обработанные изображения еще не загружены, генерируем их    
        
        
        confidence_levels = []  # Уровни уверенности  
        i=0   
        while i<0.7:
            i+=0.0001
            confidence_levels.append(i)
        # Слайдер для выбора уровня уверенности  
        slider_value = st.slider("Precision level:", 1, len(confidence_levels), 1)  
        
        res=mask()
        
        prediction=res.get_mask_n_image(model_path,st.session_state.selected_image,0)
        
        ready_image=res.mask_image(prediction["mask"],prediction["image"],0.5,100)
        
        st.image(ready_image, caption="Обработанное изображение", use_container_width=True)  # Обратите внимание на метод plot()  

        slider_value = st.slider("Выберите уровень уверенности:",   
                                 1, len(confidence_levels), 1)  
        
        result = model.predict(st.session_state.selected_image, show_boxes=False, imgsz=832, iou=1, conf=confidence_levels[slider_value - 1], max_det=1000) 
        
        
        # # Отображаем первое предсказанное изображение  
        st.write(result[0].masks)
        #st.image(result[0].save(boxes=False), caption="Обработанное изображение", use_container_width=True)  # Обратите внимание на метод plot() 

else:  
    st.warning("Пожалуйста, загрузите хотя бы одно изображение.")
