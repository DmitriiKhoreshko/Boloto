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
    # Convert the file to an opencv image.
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    st.image(opencv_image, caption="Выбранное изображение", use_container_width=True)  
    # Now do something with the image! For example, let's display it:
    res=mask()

    maskarray,result=res.get_mask("./best.pt",opencv_image,0)

    ready_image,mask1=res.get_mask_n_masked_im(maskarray,np.copy(opencv_image),0.5,100)
    im_pil = Image.fromarray(ready_image)
    st.image(result[0].plot(boxes=False), caption="Обработанное изображение", use_container_width=True)
