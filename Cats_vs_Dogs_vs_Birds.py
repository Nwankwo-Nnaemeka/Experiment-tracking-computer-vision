import os
import shutil
import random
import zipfile
import tarfile
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# To ignore some warnings about Image metadata that Pillow prints out
import warnings
warnings.filterwarnings("ignore")

cats_vs_dogs_dir = 'kagglecatsanddogs_3367a.zip'
caltech_birds_dir = 'CUB_200_2011.tar'

BASE_DIR = './tmp/data'
#os.makedirs(BASE_DIR)

# unzips the dataset
with zipfile.ZipFile(cats_vs_dogs_dir, 'r') as zip:
  zip.extractall(BASE_DIR)

with tarfile.open(caltech_birds_dir, 'r') as tar:
  tar.extractall(BASE_DIR)

base_dogs_dir = os.path.join(BASE_DIR, 'PetImages/Dog')
base_cats_dir = os.path.join(BASE_DIR,'PetImages/Cat')

raw_birds_dir = './tmp/data/CUB_200_2011/images'

base_birds_dir = os.path.join(BASE_DIR,'PetImages/Bird')
os.makedirs(base_birds_dir)

for subdir in os.listdir(raw_birds_dir):
  subdir_path = os.path.join(raw_birds_dir, subdir)
  for image in os.listdir(subdir_path):
    shutil.move(os.path.join(subdir_path, image), os.path.join(base_birds_dir))

# print the number of images for dogs, cats and birds
print(f'birds have {len(os.listdir(base_birds_dir))} images')
print(f'dogs have {len(os.listdir(base_dogs_dir))} images')
print(f'cats have {len(os.listdir(base_cats_dir))} images')

# Display an imgae from each.

image = mpimg.imread(os.path.join(base_cats_dir, os.listdir(base_cats_dir)[0]))
plt.imshow(image)
plt.show()

#image = mpimg.imread(os.path.join(base_birds_dir, os.listdir(base_birds_dir)[0]))
#plt.imshow(image)
#plt.show()

#image = mpimg.imread(os.path.join(base_dogs_dir, os.listdir(base_dogs_dir)[0]))
#plt.imshow(image)
#plt.show()
