import streamlit as st  
from PIL import Image  
from ultralytics import YOLO  # Убедитесь, что библиотека установлена  
import cv2 
from modules.mask  import *


# Заголовок приложения  
st.title("Загрузка и просмотр изображений")  



# Установка конфигурации загрузки изображений  
uploaded_file = st.file_uploader("Выберите изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=False)  
    
    
    
if uploaded_file is not None:
    st.write(type(uploaded_file))
    
    
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    st.image(opencv_image, caption="Выбранное изображение", use_container_width=True)  
    # Now do something with the image! For example, let's display it:
    res=mask()

    prediction=res.get_mask_n_image("./best.pt",opencv_image,0)

    ready_image=res.mask_image(prediction["mask"],prediction["image"],0.5,100)

    st.image(ready_image, caption="Обработанное изображение", use_container_width=True)  # Обратите внимание на метод plot()  
