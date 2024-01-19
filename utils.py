import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
import keras
from keras.models import Model
import numpy
import re
import tensorflow as tf
import tensorflow_datasets as tfds
import io
import zipfile
import logging

import os
import shutil
import random
import zipfile
import tarfile

def unzip_file(base_dir:str,  file_path:str ):

  # unzips the dataset
  for file in file_path:
    if file.endswith('.zip'):
      with zipfile.ZipFile(file, 'r') as zip:
        zip.extractall(base_dir)
    else:
      with tarfile.open(file, 'r') as tar:
        tar.extractall(base_dir)


def aggregate_files(base_dir):
  """Aggregates datasets to a single folder.
     Args:
        base_dir: the path to the directory to be processed
  """

  base_dogs_dir = os.path.join(base_dir, 'PetImages/Dog')
  base_cats_dir = os.path.join(base_dir,'PetImages/Cat')

  raw_birds_dir = '/tmp/data/CUB_200_2011/images'

  # make a directory to match that of the cats and Dogs

  base_birds_dir = os.path.join(base_dir,'PetImages/Bird')
  os.mkdir(base_birds_dir)

  # merge the directories to one
  for subdir in os.listdir(raw_birds_dir):
    subdir_path = os.path.join(raw_birds_dir, subdir)
    for image in os.listdir(subdir_path):
      shutil.move(os.path.join(subdir_path, image), os.path.join(base_birds_dir))

  return base_birds_dir, base_cats_dir, base_dogs_dir

def move_to_destination(origin, destination, percentage_split):
  num_images = int(len(os.listdir(origin))*percentage_split)
  for image_name, image_number in zip(sorted(os.listdir(origin)), range(num_images)):
    shutil.move(os.path.join(origin, image_name), destination)


def remove_zerobyte_jpg_files(directory):
 """Removes zero-byte files with a .jpg extension and non-jpg files from a directory.

 Args:
   directory: The path to the directory to be processed.
 """

 for root, _, files in os.walk(directory):
   for file in files:
     file_path = os.path.join(root, file)

     try:
       if os.path.getsize(file_path) == 0 and file.lower().endswith('.jpg'):
         os.remove(file_path)
         print(f"Removed zero-byte JPG file: {file_path}")
       elif not file.lower().endswith('.jpg'):
         os.remove(file_path)
         print(f"Removed non-JPG file: {file_path}")
     except OSError as e:
       print(f"Error removing file: {file_path} ({e})")


def plot_loss_acc(history):
  '''Plots the training and validation loss and accuracy from a history object
  Args:
      history () -- history from the models training
  '''
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(acc))

  plt.plot(epochs, acc, 'bo', label='Training accuracy')
  plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
  plt.title('Training and validation accuracy')
  plt.legend()
  plt.savefig('Accuracy.png')
  mlflow.log_artifact('Accuracy.png')
  

  plt.figure()

  plt.plot(epochs, loss, 'bo', label='Training Loss')
  plt.plot(epochs, val_loss, 'b', label='Validation Loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.savefig('Loss.png')
  mlflow.log_artifact("Loss.png")
  #plt.show()

def augment_images(image, label):
    '''Preprocess Images
    Args:
        image (array) -- pictures to be preprocessed
        label (int) -- label of the image
    '''
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(images=image, size=(300,300))
    image = image / 255.0
    return image, label


def make_model():
    '''Creates a Neural network
    Args:
    hyper_params (dictionary) -- hyperparameters for the model
    '''
    inputs = keras.layers.Input((300,300,3), dtype='float32')
    images = keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
    images = keras.layers.MaxPool2D((2,2))(images)
    images = keras.layers.Conv2D(32,(3,3), activation='relu')(images)
    images = keras.layers.MaxPool2D(2,2)(images)
    images = keras.layers.Conv2D(64,(3,3), activation='relu')(images)
    images = keras.layers.MaxPool2D(2,2)(images)
    images = keras.layers.Conv2D(128,(3,3), activation='relu')(images)
    images = keras.layers.MaxPool2D(2,2)(images)
    images = keras.layers.Flatten()(images)
    images = keras.layers.Dense(512, activation='relu')(images)
    images = keras.layers.Dense(3, activation='softmax')(images)

    model = Model(inputs, images)
    
    return model

def compile_model(model, hyper_params):
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hyper_params['learning_rate']),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy', 'loss'])
  return model

from keras.preprocessing.image import ImageDataGenerator
def preprocess_images(train_data_dir, eval_data_dir, batch_size):
  train_data_generator = ImageDataGenerator(rescale=1./255)
  val_data_generator = ImageDataGenerator(rescale=1./255)

  train_generator = train_data_generator.flow_from_directory(train_data_dir,
                                                             target_size=(300,300),
                                                             batch_size=batch_size,
                                                             class_mode='sparse')
  
  validation_generator = val_data_generator.flow_from_directory(eval_data_dir,
                                                                target_size=(300,300),
                                                                batch_size=batch_size,
                                                                class_mode='sparse')
  
  return train_generator, validation_generator
  
def infer_sample_signature(dir):
  train_data_generator = ImageDataGenerator(rescale=1./255)
  sample_input = train_data_generator.flow_from_directory(
    directory = dir,  
    target_size=(300, 300),
    batch_size=1, 
    class_mode="sparse" 
).next()
  return sample_input 
  