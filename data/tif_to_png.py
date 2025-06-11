import os
from PIL import Image

def convert_tif_to_png(folder_path):
    """
    Converts all .tif or .tiff images in the specified folder to .png format.
    The output file will have the same name, with the .png extension.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.tif', '.tiff')):
            tif_path = os.path.join(folder_path, filename)
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(folder_path, png_filename)
            try:
                with Image.open(tif_path) as img:
                    img.save(png_path, 'PNG')
                print(f"Converted: {filename} -> {png_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")


convert_tif_to_png('data/hq_images/')