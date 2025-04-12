import os
from dotenv import load_dotenv
import ast

load_dotenv()

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
LEARNING_RATE = float(os.getenv("LR_RATE", 0.001))
EPOCHS = int(os.getenv("EPOCHS", 20))
SEED = int(os.getenv('SEED', 1))

#Dataset Configs
IM_SIZE = int(os.getenv("IM_SIZE", 224))
COLOR_RANGE = int(os.getenv("COLOR_RANGE", 255))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 3))
CLASS_NAME_str = os.getenv('CLASS_NAME',['angry', 'happy', 'sad'])
CLASS_NAME = ast.literal_eval(CLASS_NAME_str)

#Layers
DENSE1 = int(os.getenv("DENSE1", 128))
DENSE2 = int(os.getenv("DENSE2", 32))
ACTIVATION1 = os.getenv("ACTIVATION1")
ACTIVATION2 = os.getenv("ACTIVATION2")

#Base Model
WEIGHTS = os.getenv("WEIGHTS")
CHANNELS = int(os.getenv("CHANNELS"))