import os
import random
from PIL import Image, ImageEnhance
import numpy as np

# Paths
input_folder = 'data/hq_images/'           # Folder with 1000 high-quality images
output_folder = 'data/low_light_images/'   # Output folder for 8000 synthetic images

os.makedirs(output_folder, exist_ok=True)

# Enhancement parameters
gamma_values = [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2]
brightness_factors = [0.3, 0.4, 0.5, 0.6]

def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    return image.point(table * 3)

def create_low_light_versions(img, image_name, idx):
    for i in range(8):  # 8 variations per image
        gamma = random.choice(gamma_values)
        brightness = random.choice(brightness_factors)

        dark_img = adjust_gamma(img, gamma)
        enhancer = ImageEnhance.Brightness(dark_img)
        low_light = enhancer.enhance(brightness)

        out_name = f"{os.path.splitext(image_name)[0]}_ll_{i}.png"
        low_light.save(os.path.join(output_folder, out_name))

# Process images
for idx, image_name in enumerate(os.listdir(input_folder)):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
        img_path = os.path.join(input_folder, image_name)
        img = Image.open(img_path).convert("RGB")
        create_low_light_versions(img, image_name, idx)
