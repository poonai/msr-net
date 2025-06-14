import glob, os
from torchvision.io import decode_image
from pathlib import Path
import torch

img_dir = 'data/'
x_dir = Path(os.path.join(img_dir, "hq_images_patches"))
x_img_path = [file for file in x_dir.glob("*") if file.is_file()]

for idx, x_img in enumerate(x_img_path):
    
    if idx%10000:
        print(f'scanning progress {idx/len(x_img_path) *100} current:{x_img}')
        
    try:
        decode_image(x_img)
    except Exception as e:
        print(f'found corrupted img {x_img} {e}')
        x_splits = x_img.name.split("_")
        print(f'removing src img {x_splits[0]}')
        for f in glob.glob(f'data/hq_images_patches/{x_splits[0]}_*.png'):
            print(f'removing hq patch {f}')
            os.remove(f)
            
        for f in glob.glob(f'data/low_light_images_patches/{x_splits[0]}_*.png'):
            print(f'removing low_light patch {f}')
            os.remove(f)
        