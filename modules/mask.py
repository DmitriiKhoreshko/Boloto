import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image  

class mask:
    def get_mask(self, model_path , image, conf):
        """Получаем на выход маску"""
        model = YOLO(model_path)
        
        result = model.predict(image, show_boxes=False, imgsz=832, iou=1, conf=conf, max_det=250)
        
        return result[0].masks.xy

    def get_mask_n_masked_im(self, masks, image, transparensy,trash):  
            
            transparency = transparensy  # Уровень прозрачности (значение от 0 до 1)  
            alpha_value = int(255 * transparency)
            
            b_mask = np.zeros((image.shape[0], image.shape[1], 4), np.uint8) 
            for mask in masks:
                try:
                    if mask.shape[0]>=trash:
                        
                        contour = mask
                        
                        contour = contour.astype(int)
                        contour = contour.reshape(-1, 1, 2)
                        
                        b_mask = cv2.fillPoly(b_mask, [contour], (0, 255, 0))  
                    else:
                        continue
                except:
                    pass
            alpha_channel = b_mask[:, :, 3] 
            alpha_channel[alpha_channel > 0] = alpha_value 
            alpha_channel[alpha_channel == 0] = 0
 
            b_mask_rgb = b_mask[:, :, :3]
            isolated = cv2.addWeighted(image, 1, b_mask_rgb, transparency, 0)
            return isolated, b_mask_rgb
