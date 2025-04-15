import os
import sys
import math
import numpy
import tensorflow as tf

config_path = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(config_path)
from configurations import config

class DatasetUtil:
    def prepare_data(self, path_to_directory: str):
        dataset = tf.keras.utils.image_dataset_from_directory(
            path_to_directory,
            batch_size = config.BATCH_SIZE,
            image_size = (config.IM_SIZE, config.IM_SIZE),
            seed = config.SEED,
            class_names = config.CLASS_NAME,
            label_mode = 'categorical'
        )

        prepared_dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return prepared_dataset
    

    def resize_rescale(self, image, label):
        resize = tf.image.resize(image, (config.IM_SIZE, config.IM_SIZE))
        rescale = resize/255.
        return rescale, label
    
    def load_single_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=config.CHANNELS)
        return image
    
    def get_absolute_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(os.getcwd(), path)






def scheduler(epochs, lr):
    if epochs == 1:
        return lr
    else:
        return (lr*math.exp(-0.1))

def get_base_model(model_name: str) -> tf.keras.Model:
    # try:
    #     base_model = getattr(tf.keras.applications, model_name)
    #     base_model(include_top=False, weights=config.WEIGHTS,\
    #                 input_shape=(config.IM_SIZE, config.IM_SIZE, config.CHANNELS))
    # except:
    base_model = tf.keras.applications.ResNet50V2(include_top=False, weights=config.WEIGHTS,\
                                        input_shape=(config.IM_SIZE, config.IM_SIZE, config.CHANNELS))
    base_model.trainable=False
    return base_model
