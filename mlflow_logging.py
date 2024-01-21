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

BASE_DIR = './tmp/data'
train_data_dir = os.path.join(BASE_DIR, 'train')
eval_data_dir = os.path.join(BASE_DIR, 'eval')

hyper_parameters = {'learning_rate':0.001, 'batch_size': 32, 'drop_out': None, 'epochs':5}

train_generator, validation_generator = preprocess_images(train_data_dir, eval_data_dir, hyper_parameters['batch_size'])

model = create_and_compile_model(hyper_parameters)

with mlflow.start_run() as run:
        model, history = fit_model(model, train_generator,validation_generator,
                  hyper_parameters )
        mlflow.log_params(hyper_parameters)
        #mlflow.log_metrics(metrics=metrics)

        plot_loss_acc(history)

        mlflow.log_artifact('accuracy.png')
        mlflow.log_artifact('loss.png')
        # Log model
        sample_input = infer_sample_signature(train_data_dir)
        signature = infer_signature(sample_input, model.predict(sample_input))

        mlflow.tensorflow.log_model(model, artifact_path = "model", signature=signature)

        model_uri = run.info.artifact_uri + "/model"



# Using image Documentation


#predictions = loaded_model.predict()
#print(predictions)