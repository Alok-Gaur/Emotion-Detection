import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
LEARNING_RATE = float(os.getenv("LR_RATE", 0.001))
EPOCHS = int(os.getenv("EPOCHS", 20))
IM_SIZE = int(os.getenv("IM_SIZE", 224))
SEED = int(os.getenv('SEED', 1))
CLASS_NAME = os.getenv('CLASS_NAME',['angry', 'happy', 'sad'])