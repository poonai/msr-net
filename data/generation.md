# Steps to create dataset for training

- create `hq_images` and `low_light_images` folder in data directory
- download uidc dataset from [Kaggle](https://www.kaggle.com/datasets/flamense160/ucid-dataset).
- extract the uidc dataset and paste the source images in `hq_images` folder.
- I'm using `uv` to as package manager and suggest to use the same. Othewise, figure yourself 
  on installing the relevant dependencies.
- execute `uv run data/tif_to_png.py` to transform tif file to png. I'm doing this step only 
  because torch needs png file.
- execute `uv run data/generate.py` from the root folder. It will result in generating all the 
  low light images of hq_images.
- execute `uv run data/patch_extracter.py` to create fixed size patch from the low light and hq
  images.
