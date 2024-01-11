import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
import keras
from keras.models import Model

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


def make_model(hyper_params):
    '''Creates a Neural network
    Args:
    hyper_params (dictionary) -- hyperparameters for the model
    '''
    inputs = keras.Input((300,300), dtype='float32')
    images = keras.layers.Conv2D(16, (3,3), activation='relu')(inputs)
    images = keras.layers.MaxPool2D((2,2))(images)
    images = keras.layers.Conv2D(32,(3,3), activation='relu')(images)
    images = keras.layers.MaxPool2D(2,2)(images)
    images = keras.layers.Conv2D(64,(3,3), activation='relu')(images)
    images = keras.layers.MaxPool2D(2,2)(images)
    images = keras.layers.Flatten()(images)
    images = keras.layers.Dense(512, activation='relu')(images)
    images = keras.layers.Dense(1, activation='sigmoid')

    model = Model(inputs, images)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hyper_params['learning_rate']),
                  loss='binary_crossentropy',
                   metrics=['accuracy', 'loss'])
    
    return model
