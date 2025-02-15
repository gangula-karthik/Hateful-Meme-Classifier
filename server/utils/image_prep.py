import easyocr as eo
import cv2
from io import BytesIO
import numpy as np
import base64
import json
import os
from PIL import Image
from tqdm import tqdm


def extract_text_with_binary_mask(image_path, languages=['en'], gpu=False):
    if "base64," in image_path:
        base64_data = image_path.split("base64,")[1]
    else:
        base64_data = image_path

    image_data = base64.b64decode(base64_data)
    image = np.array(Image.open(BytesIO(image_data)))


    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # RGBA image
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    reader = eo.Reader(languages, gpu=gpu)
    results = reader.readtext(image)
    
    height, width, _ = image.shape
    
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    
    for result in results:
        bbox, text, score = result
        l_bbox, l_bbox1 = bbox[0]
        r_bbox, r_bbox1 = bbox[2]
        
        cv2.rectangle(binary_mask, 
                      (int(l_bbox), int(l_bbox1)), 
                      (int(r_bbox), int(r_bbox1)), 
                      255, -1)
    
    return image, binary_mask

def image_inpainting(image, mask): 
    text_removed_image = cv2.inpaint(image, mask, inpaintRadius=0.03, flags=cv2.INPAINT_TELEA)
    res = cv2.cvtColor(text_removed_image, cv2.COLOR_BGR2RGB)
    return res

def resize_and_center_crop(image, target_min=300, target_max=512, target_size=512):
    height, width, _ = image.shape

    if width < target_min or height < target_min:
        scale_factor = target_min / min(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = cv2.resize(image, (new_width, new_height))

    height, width, _ = image.shape
    if width > target_max or height > target_max:
        scale_factor = target_max / max(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = cv2.resize(image, (new_width, new_height))

    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2
    crop_x1 = max(center_x - target_size // 2, 0)
    crop_x2 = min(center_x + target_size // 2, width)
    crop_y1 = max(center_y - target_size // 2, 0)
    crop_y2 = min(center_y + target_size // 2, height)
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    if cropped_image.shape[0] != target_size or cropped_image.shape[1] != target_size:
        cropped_image = cv2.resize(cropped_image, (target_size, target_size))

    return cropped_image


def image_processing_pipeline(image): 
    img, mask = extract_text_with_binary_mask(image)
    res_img = image_inpainting(img, mask)
    res_img = resize_and_center_crop(res_img)
    return res_img


# if __name__ == "__main__":

#     def encode_image(image_path):
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read())
    
#     res_string = encode_image('Image_16.jpg')
#     res = image_processing_pipeline(res_string)
#     print(res)
