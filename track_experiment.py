import os
import sys
import shutil

def main():
    if len(sys.argv) != 2:
        print("Usage: python move_pngs.py <experiment_id>")
        sys.exit(1)

    experiment_id = sys.argv[1]
    target_dir = f"experiment_{experiment_id}"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Only list PNGs in the current directory (not subdirectories)
    png_files = [f for f in os.listdir('.') if os.path.isfile(f) and f.lower().endswith('.png')]
    if not png_files:
        print("No PNG files found in the current directory.")
        return

    for file in png_files:
        shutil.move(file, os.path.join(target_dir, file))
        print(f"Moved {file} to {target_dir}/")

    print(f"All PNG files moved to {target_dir}/")

if __name__ == "__main__":
    main()