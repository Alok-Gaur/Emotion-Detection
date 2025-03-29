import os
import sys
import tensorflow as tf

config_path = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(config_path)
from configurations import config

class DatasetUtil:
    def resize_rescale(self, image, label):
        resize = tf.image.resize(image, (config.IM_SIZE, config.IM_SIZE))
        rescale = resize/255.
        return rescale, label

