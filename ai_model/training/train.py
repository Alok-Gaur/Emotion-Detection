import os
import sys
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
from ai_model.training.emotion_detection_model import EmotionDetection
from ai_model.training.util import DatasetUtil, scheduler

#Import from local environment
config_path = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(config_path)

import configurations.config as config


def train_model(train_directory_path: str, val_directory_path:str):
    #Get the absolute path of the directories
    data_obj = DatasetUtil()
    train_directory_path = data_obj.get_absolute_path(train_directory_path)
    val_directory_path = data_obj.get_absolute_path(val_directory_path)

    #Load the Dataset
    training_dataset = data_obj.prepare_data(train_directory_path)
    validation_dataset = data_obj.prepare_data(val_directory_path)


    # Resizing and rescaling the dataset(Optional)
    # train_dataset = train_dataset.map(lambda img, label: DatasetUtil.resize_rescale(img, label))
    # val_dataset = val_dataset.map(lambda img, label: DatasetUtil.resize_rescale(img, label))


    #Initialize the Model
    model = EmotionDetection(config.MODEL_NAME, config.NUM_CLASSES, resized=False, rescaled=False)

    #Setup the Callbacks
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('ai_model/model/Emotion_Detection4.weights.h5',\
                                                    save_best_only=True, save_weights_only=True)

    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE),
                loss=CategoricalCrossentropy(),
                metrics=[CategoricalAccuracy(name='Accuracy'), TopKCategoricalAccuracy(k=2, name="TopKAccuracy")])


    history = model.fit(training_dataset.take(150), validation_data=validation_dataset.take(40),\
                        epochs=config.EPOCHS, verbose=1, callbacks=[scheduler_callback, checkpoint])

    model.save("ai_model/model/emotion_detection_model4.keras")