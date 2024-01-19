import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds 
import keras
from keras.models import Model
import mlflow
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import Schema, TensorSpec
import cv2
import matplotlib.pyplot as plt
from utils import *


# Use to activate the  UI server
# mlflow server --host 127.0.0.1 --port 8080
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Run the tracking server: mlflow ui --port 5000
# export MLFLOW_TRACKING_URI=http://localhost:5000


mlflow.set_experiment("Cats_vs_Dogs Experiment")

#setattr(tfds.image_classification.cats_vs_dogs, 
      #  '_URL',
    #    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

tfds.image_classification.cats_vs_dogs.CatsVsDogs._generate_examples = _generate_examples

train_data = tfds.load('cats_vs_dogs', split='train[:80%]', data_dir='Datasets/training_dir', as_supervised=True, download = True)
test_data = tfds.load('cats_vs_dogs', split='train[80%:90%]', data_dir='Datasets/test_dir', as_supervised=True)
validation_data= tfds.load('cats_vs_dogs', split='train[-10%:]', data_dir='Datasets/validation_dir', as_supervised=True)



hyper_parameters = {'learning_rate':0.001, 'batch_size': 32, 'drop_out': None, 'epochs':5}

# Option 1: manually construct the signature object
input_schema = Schema(
    [
        TensorSpec(np.dtype(np.float32), (-1, 300, 300))
    ]
)

output_schema = Schema(
    [
        TensorSpec(np.dtype(np.float32), (-1, 1))
        ]
        )

signature = ModelSignature(inputs=input_schema, outputs=output_schema)
# Option 2: Infer the signature
# signature = infer_signature(train_data[0], train_data[1])


augmented_train_data = train_data.map(augment_images)
train_batches = augmented_train_data.shuffle(1024).batch(hyper_parameters['batch_size'])
augmented_valid_data = validation_data.map(augment_images)
validation_batches = augmented_valid_data.batch(hyper_parameters['batch_size'])
augmented_test_data = test_data.map(augment_images)
test_batches = augmented_test_data.batch(hyper_parameters['batch_size'])


model = make_model()

def run_model(hyper_params, training, validation, testing):
    model = make_model(hyper_params)

    
    
    with mlflow.start_run() as run:

        history = model.fit(training, 
                  validation_data=validation,
                   batch_size=hyper_params['batch_size'], 
                   epochs=hyper_params['epochs'],)
        
        results = model.evaluate(testing)
        #print(model.metrics_names)

        binary_crossentropy = results[0]
        accuracy = results[1]
        loss = results[2]
        metrics = {'BCE': binary_crossentropy, 'accuracy': accuracy, 'loss': loss}

        mlflow.log_params(hyper_params)
        mlflow.log_metrics(metrics=metrics)

        plot_loss_acc(history)

        # Log model
        mlflow.tensorflow.log_model(model, artifact_path = "model", signature=signature)

        model_uri = run.info.artifact_uri + "/model"



        return model, model_uri

model, model_uri = run_model(hyper_parameters, train_batches, validation_batches, test_batches)

loaded_model = mlflow.tensorflow.load_model(model_uri)

predictions = loaded_model.predict(test_batches[0])
print(predictions)