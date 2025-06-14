import os
import random
from PIL import Image, ImageEnhance
import modal
from pathlib import Path

# Paths
input_folder = '/root/data/data/hq_images/'           # Folder with 1000 high-quality images
output_folder = '/root/data/data/low_light_images/'   # Output folder for 8000 synthetic images



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
        
app = modal.App("msr-net-generator")
volume = modal.Volume.from_name("msr-net-data", create_if_missing=False)

base_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pydantic==2.9.1",
    "pillow>=11.2.1",
)

def create_patches(input_folder, output_folder, patch_size=64):
    """
    Create 64x64 patches from all images in input_folder and save to output_folder.
    
    Args:
        input_folder (str): Path to folder containing original images
        output_folder (str): Path to save patches
        patch_size (int): Size of square patches (default: 64)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    
    # Process each image in input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            try:
                # Open image
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path)
                img_name = os.path.splitext(filename)[0]
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                width, height = img.size
                
                # Skip if image is smaller than patch size
                if width < patch_size or height < patch_size:
                    print(f"Skipping {filename} - smaller than {patch_size}x{patch_size}")
                    continue
                
                # Calculate number of patches
                num_patches_x = width // patch_size
                num_patches_y = height // patch_size
                
                # Crop to exact multiple of patch size
                cropped_width = num_patches_x * patch_size
                cropped_height = num_patches_y * patch_size
                img = img.crop((0, 0, cropped_width, cropped_height))
                
                # Extract and save patches
                patch_count = 0
                for i in range(num_patches_y):
                    for j in range(num_patches_x):
                        left = j * patch_size
                        upper = i * patch_size
                        right = left + patch_size
                        lower = upper + patch_size
                        
                        patch = img.crop((left, upper, right, lower))
                        
                        # Save patch with position info
                        patch_name = f"{img_name}_patch_{patch_count}.png"
                        patch.save(os.path.join(output_folder, patch_name))
                        patch_count += 1
                
                print(f"Created {patch_count} patches from {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


with base_image.imports():
    import os
    import random
    from PIL import Image, ImageEnhance

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

@app.function(image=base_image,timeout=10*HOURS,volumes={"/root/data":volume})
def generate():
    folders = [('/root/data/data/hq_images', '/root/data/data/hq_images_patches'),
          ('/root/data/data/low_light_images', '/root/data/data/low_light_images_patches')]
    #os.makedirs(output_folder, exist_ok=True)
    # Process images
    for folder in folders:
     create_patches(folder[0], folder[1])
    # for idx, image_name in enumerate(os.listdir(input_folder)):
    #     if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
    #         img_path = os.path.join(input_folder, image_name)
    #         img = Image.open(img_path).convert("RGB")
    #         create_low_light_versions(img, image_name, idx)

@app.local_entrypoint()
def main():
    generate.remote()