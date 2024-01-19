from utils import *
import mlflow
import numpy as np
from mlflow.types.schema import Schema, TensorSpec
from mlflow.models import ModelSignature, infer_signature

# Use to activate the  UI server
# mlflow server --host 127.0.0.1 --port 8080
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Run the tracking server: mlflow ui --port 5000
# export MLFLOW_TRACKING_URI=http://localhost:5000

# Set experiment name
mlflow.set_experiment("Cats_vs_Dogs Experiment")


hyper_parameters = {'learning_rate':0.001, 'batch_size': 32, 'drop_out': None, 'epochs':5}

model = make_model()
# model.summary()
model = compile_model(model, hyper_parameters)

def run_model(hyper_params, training, validation, plot_image, dir_path_infer):
    """"""
    
    with mlflow.start_run() as run:

        history = model.fit(training, 
                  validation_data=validation,
                   batch_size=hyper_params['batch_size'], 
                   epochs=hyper_params['epochs'],)

        mlflow.log_params(hyper_params)
        #mlflow.log_metrics(metrics=metrics)

        plot_loss_acc(history)

        mlflow.log_artifact(plot_image[0])
        mlflow.log_artifact(plot_image[1])
        # Log model
        sample_input = infer_sample_signature(dir_path_infer)
        signature = infer_signature(sample_input, model.predict(sample_input))

        mlflow.tensorflow.log_model(model, artifact_path = "model", signature=signature)

        model_uri = run.info.artifact_uri + "/model"



        return model, model_uri

# Using image Documentation
BASE_DIR = './tmp/data'
train_data_dir = os.path.join(BASE_DIR, 'train')
eval_data_dir = os.path.join(BASE_DIR, 'eval')

train_generator, validation_generator = preprocess_images(train_data_dir, eval_data_dir, batch_size=hyper_parameters['batch_size'])

model, model_uri = run_model(hyper_parameters, train_generator, validation_generator, ('Acuuracy.png', 'Loss.png'), train_data_dir)

loaded_model = mlflow.tensorflow.load_model(model_uri)

predictions = loaded_model.predict()
print(predictions)