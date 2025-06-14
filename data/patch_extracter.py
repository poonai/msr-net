import os
from PIL import Image

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

# folders = [('data/hq_images', 'data/hq_images_patches'),
#           ('data/low_light_images', 'data/low_light_images_patches')]
# for folder in folders:
#     create_patches(folder[0], folder[1])