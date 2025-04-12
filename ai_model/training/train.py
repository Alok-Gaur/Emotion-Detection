import os
import sys
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
from emotion_detection_model import EmotionDetection
from util import DatasetUtil, scheduler

#Import from local environment
config_path = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(config_path)

import configurations.config as config

#initialize the Dataset Directories
train_directory = os.path.join(os.getcwd(), 'ai_model', 'data', 'train')
val_directory = os.path.join(os.getcwd(), 'ai_model', 'data', 'test')

#Load the Dataset
data_obj = DatasetUtil()
training_dataset = data_obj.prepare_data(train_directory)
validation_dataset = data_obj.prepare_data(val_directory)


# Resizing and rescaling the dataset(Optional)
# train_dataset = train_dataset.map(lambda img, label: DatasetUtil.resize_rescale(img, label))
# val_dataset = val_dataset.map(lambda img, label: DatasetUtil.resize_rescale(img, label))


#Initialize the Model
model = EmotionDetection('ResNet50V2', config.NUM_CLASSES, resized=False, rescaled=False)

#Setup the Callbacks
scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint('Emotion_Detection4.weights.h5',\
                                                save_best_only=True, save_weights_only=True)

model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE),
              loss=CategoricalCrossentropy(),
              metrics=[CategoricalAccuracy(name='Accuracy'), TopKCategoricalAccuracy(k=2, name="TopKAccuracy")])


history = model.fit(training_dataset.take(150), validation_data=validation_dataset.take(40),\
                    epochs=config.EPOCHS, verbose=1, callbacks=[scheduler_callback, checkpoint])

model.save("ai_model/model/emotion_detection_model4.keras")