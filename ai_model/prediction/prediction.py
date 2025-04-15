import os
import sys
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from ai_model.training.util import DatasetUtil
from typing import List

config_path = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(config_path)

from configurations import config
def make_prediction(image_path:str, directory_path:str, model_path: str) -> List[str]:
    #if a single image is given it will only predict and show the result
    #if a folder is given we assume that it is on proper structure
    #as mentioned on the readme file.
    data_obj = DatasetUtil()
    try:
        if image_path:
            absolute_image_path = data_obj.get_absolute_path(image_path)
            test_images = data_obj.load_single_image(absolute_image_path)
        else:
            absolute_directory_path = data_obj.get_absolute_path(directory_path)
            test_dataset = data_obj.prepare_data(path_to_directory=absolute_directory_path)

            images_batch, label_batch = zip(*test_dataset)

            test_images = tf.concat(images_batch, axis=0)
            labels = tf.concat(label_batch, axis=0)
    except Exception as e:
        print(e)

    model = tf.keras.models.load_model(data_obj.get_absolute_path(model_path))
    history = model.predict(test_images)

    prediction = []
    for pred in history:
        pred_label_index = tf.argmax(pred)
        prediction.append(config.CLASS_NAME[pred_label_index])
    
    return prediction