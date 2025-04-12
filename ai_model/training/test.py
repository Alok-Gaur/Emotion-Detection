import os
import sys
import tensorflow as tf
from emotion_detection_model import EmotionDetection
from tensorflow.keras.applications import ResNet50V2
from util import DatasetUtil

config_path = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(config_path)

from configurations import config

test_directory = os.path.join(os.getcwd(), 'ai_model', 'data', 'test')

data_obj = DatasetUtil()
test_dataset = data_obj.prepare_data(path_to_directory=test_directory)

images_batch, label_batch = zip(*test_dataset.take(3))

test_images = tf.concat(images_batch, axis=0)
labels = tf.concat(label_batch, axis=0)

# base_model = ResNet50V2(include_top=False, weights=config.WEIGHTS,/
#                         input_shape=(config.IM_SIZE, config.IM_SIZE, config.CHANNELS))
# model = EmotionDetection(base_model=base_model, num_classes=config.NUM_CLASSES, resized=False, rescaled=False)
# model.load_weights("Emotion_Detection2")

model = tf.keras.models.load_model("C:/Users/user/Desktop/Emotion Detection/ai_model/model/emotion_detection_model4.keras")
history = model.predict(test_images)

print("predictions - actual")
for pred_label, true_label in zip(history, labels):
    pred = tf.argmax(pred_label)
    actual = tf.argmax(true_label)

    print(f"{config.CLASS_NAME[pred]} - {config.CLASS_NAME[actual]}")