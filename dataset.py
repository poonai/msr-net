from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import decode_image

import os

class LowLightDataset(Dataset):
    def __init__(self, img_dir = 'data/'):
        x_dir = Path(os.path.join(img_dir, 'low_light_images_patches'))
        self.x_img_path = [file for file in x_dir.glob('*') if file.is_file()]
        self.img_dir = img_dir
        
        
    def __len__(self):
        return len(self.x_img_path)
    
    def __getitem__(self, idx):
        x_path = self.x_img_path[idx]
        
        x_splits = x_path.name.split('_')
        y_file_name = f'{x_splits[0]}_patch_{x_splits[len(x_splits)-1]}'
        y_path = os.path.join(self.img_dir, 'hq_images_patches', y_file_name)
        return decode_image(x_path), decode_image(y_path)
        
        
        
        