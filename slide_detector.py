from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing
import os
import shutil

IMG_SIZE = 150
# batch size should probably be smaller...
batch_size = 64
num_classes = 2
train_set_ratio = 0.8
val_set_ratio = 0.1
test_set_ratio = 0.1

