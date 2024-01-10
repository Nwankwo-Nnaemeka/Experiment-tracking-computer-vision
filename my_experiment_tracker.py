import tensorflow as tf
import numpy as np 
import keras
import tensorflow_datasets as tfds 
from keras.models import Model, Sequential
import cv2
import mlflow
from mlflow.models import ModelSignature, infer_signature
import matplotlib.pyplot as plt
from plot_utils import plot_loss_acc

# Use to activate the  UI server
# mlflow server --host 127.0.0.1 --port 8080
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Run the tracking server: mlflow ui --port 5000
# export MLFLOW_TRACKING_URI=http://localhost:5000


mlflow.set_experiment("Cats_vs_Dogs Experiment")

train_data = tfds.load('cats_vs_dogs', split='train[:80%]', data_dir='Datasets/training_dir', as_supervised=True)
test_data = tfds.load('cats_vs_dogs', split='train[80%:90%]', data_dir='Datasets/test_dir', as_supervised=True)
validation_data= tfds.load('cats_vs_dogs', split='train[-10%:]', data_dir='Datasets/validation_dir', as_supervised=True)

def augment_images(image, label):
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(images=image, size=(300,300))
    image = image / 255.0
    return image, label

hyper_parameters = {'learning_rate':0.001, 'batch_size': 32, 'drop_out': None, 'epochs':5}

augmented_train_data = train_data.map(augment_images)
train_batches = augmented_train_data.shuffle(1024).batch(hyper_parameters['batch_size'])
augmented_valid_data = validation_data.map(augment_images)
validation_batches = augmented_valid_data.batch(hyper_parameters['batch_size'])
augmented_test_data = test_data.map(augment_images)
test_batches = augmented_test_data.batch(hyper_parameters['batch_size'])

signature = infer_signature(train_data[0], train_data[1])

def make_model(hyper_params):
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

model = make_model()


print(model.summary())

def run_model(hyper_params, training, validation, testing):
    model = make_model(hyper_params)

    
    
    with mlflow.start_run() as run:

        history = model.fit(training, 
                  validation_data=validation,
                   batch_size=hyper_params['batch_size'], 
                   epochs=hyper_params['epochs'],)
        
        results = model.evaluate(testing)
        print(model.metrics_names)

        binary_crossentropy = results[0]
        accuracy = results[1]
        loss = results[2]
        metrics = {'BCE': binary_crossentropy, 'accuracy': accuracy, 'loss': loss}

        mlflow.log_params(hyper_params)
        mlflow.log_metrics(metrics=metrics)

        plot_loss_acc(history)
        # Log model
        mlflow.tensorflow.log_model(model, "model", signature=signature)



        return model

model = run_model(hyper_parameters, train_batches, validation_batches, test_batches)
