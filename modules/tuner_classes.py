import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras_tuner import HyperModel

from utils.loss_and_scoring import FocalLoss, f1_score


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])

preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

IMG_SIZE = (300, 300)

class CustomHyperModel(HyperModel):
    def build(self, hp):
        num_classes = 2
        base_model = tf.keras.applications.EfficientNetV2B3(
            include_top=False, 
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
        base_model.trainable = False

        inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)

        num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=4, step=1)
        
        for i in range(num_dense_layers):
            x = layers.Dense(units=hp.Int(f'dense_units_{i}', min_value=64, max_value=512, step=64), activation='relu')(x)
        
        x = layers.Dropout(hp.Float(f'dropout', min_value=0.2, max_value=0.25, step=0.1))(x)

        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [0.01, 0.001, 0.0001])),
            loss=FocalLoss(),
            metrics=['accuracy', f1_score]
        )
        return model
    

class HyperparameterLogger(keras.callbacks.Callback):
    def __init__(self, log_file='hyperparameters_log.csv'):
        super(HyperparameterLogger, self).__init__()
        self.log_file = log_file
        self.log_data = []

    def on_trial_end(self, trial):
        metrics = trial.oracle.get_trial(trial.trial_id).metrics
        hp_values = {hp.name: hp.value for hp in trial.hyperparameters}
        entry = {**hp_values, **metrics}
        self.log_data.append(entry)

        df_logs = pd.DataFrame(self.log_data)
        df_logs.to_csv(self.log_file, index=False)

    def on_batch_end(self, batch, logs=None):
        pass
