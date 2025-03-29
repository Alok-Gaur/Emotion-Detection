import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Resizing, Rescaling, Dropout, Conv2D, MaxPool2D, Normalization, BatchNormalization, \
                                    flatten, Dense, RandomFlip, RandomRotation, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import  CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
from dataset_util import DatasetUtil
# Import from local environment
config_path = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(config_path)

import configurations.config as config

#initialize the Dataset Directories
train_directory = os.path.join(os.getcwd(), 'ai_model', 'data', 'train')
val_directory = os.path.join(os.getcwd(), 'ai_model', 'data', 'test')

#Load the Dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    batch_size = config.BATCH_SIZE,
    image_size = (config.IM_SIZE, config.IM_SIZE),
    seed = config.SEED,
    class_name = config.CLASS_NAME,
    label_mode = 'categorical'
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_directory,
    batch_size = config.BATCH_SIZE,
    image_size = (config.IM_SIZE, config.IM_SIZE),
    seed = config.SEED,
    class_name = config.CLASS_NAME,
    label_mode = 'categorical'
)

# Resizing and rescaling the dataset
train_dataset = train_dataset.map(lambda img, label: DatasetUtil.resize_rescale(img, label))
val_dataset = val_dataset.map(lambda img, label: DatasetUtil.resize_rescale(img, label))
