import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Resizing, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable
from util import get_base_model

config_path = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(config_path)

from configurations import config

@register_keras_serializable(package='Custom')
class EmotionDetection(Model):
    def __init__(self, base_model_name: str, num_classes=2, resized=False, rescaled=False):
        super(EmotionDetection, self).__init__()
        # Pre-trained Model
        self.base_model_name = base_model_name
        self.base_model = get_base_model(self.base_model_name)
        
        #Essential Parameters
        self.num_classes = num_classes
        self.resized = resized
        self.rescaled = rescaled
        
        #Layers
        self.preprocessing = self._preprocessing()
        self.flatten = Flatten()
        self.batch_norm = BatchNormalization()
        self.dense1 = Dense(config.DENSE1, activation=config.ACTIVATION1)
        self.dense2 = Dense(config.DENSE2, activation=config.ACTIVATION1)
        self.result = Dense(self.num_classes, activation=config.ACTIVATION2)

    
    def _preprocessing(self):
        preprocessing = tf.keras.Sequential(name='preprocessing')
        if not self.resized:
            preprocessing.add(Resizing(config.IM_SIZE, config.IM_SIZE))
        if not self.rescaled:
            preprocessing.add(Rescaling(1./config.COLOR_RANGE))
        return preprocessing

    def call(self, input, training=False):
        x = self.preprocessing(input)
        x = self.base_model(input)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batch_norm(x)
        x = self.dense2(x)
        return self.result(x)
    
    def compile(self, loss, optimizer, metrics):
        super(EmotionDetection, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.loss_obj = loss
        self.optimizer_obj = tf.keras.optimizers.get(optimizer)
        self.metrics_obj = [tf.keras.metrics.get(metric) for metric in metrics]
    
    def train_step(self, inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            prediction = self(images, training=True)
            loss = self.loss_obj.call(labels, prediction) 

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer_obj.apply_gradients(zip(gradients, self.trainable_variables))

        for metric in self.metrics_obj:
            metric.update_state(labels, prediction)
        
        return {'loss': loss, **{metric.name: metric.result() for metric in self.metrics_obj}}
    
    def test_step(self, inputs):
        images, labels = inputs
        prediction = self(images, training=False)
        loss = self.loss_obj(labels, prediction)

        for metric in self.metrics_obj:
            metric.update_state(labels, prediction)
        
        return {'loss': loss, **{metric.name: metric.result() for metric in self.metrics_obj}}
    
    
    #Initializes the Class Attributes
    #Initialize the __init__ methods Parameters which is 
    #essesntial when loading the model from the disk
    def get_config(self):
        return {
                'base_model_name': self.base_model_name,
                'num_classes': self.num_classes,
                'resized': self.resized,
                'rescaled': self.rescaled
                }
    
    #Load the model based on the parameters set by the
    #get_config method
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    