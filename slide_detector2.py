from tensorflow.keras.applications import EfficientNetB0 as Net
# from tensorflow.keras.applications import center_crop_and_resize, preprocess_input

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import losses
from tensorflow.keras import metrics
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_labeler import prepare_for_training


batch_size = 2

width = 150
height = 150
epochs = 20
validation_steps = 20
input_shape = (height, width, 3)
train_dir = os.path.join(os.curdir, 'train_dir')

x, y = prepare_for_training(train_dir)
split = 0.1
te_index = int(x.shape[0] * (split))
teX, teY = x[te_index: te_index*2], y[te_index: te_index*2]
trX, trY = x[te_index*2:], y[te_index*2:]
vX, vY = x[0:te_index], y[0:te_index]

print(len(teX), len(teY))
print(len(trX), len(trY))
print(len(vX), len(vY))

def build_model():
    # loading pretrained conv base model
    conv_base = Net(weights="imagenet", include_top=False, input_shape=input_shape)

    dropout_rate = 0.2
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D(name="gap"))
    model.add(layers.Flatten(name="flatten"))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="dropout_out"))
    model.add(layers.Dense(1, activation="sigmoid", name="bin_out"))

    conv_base.trainable = False

    model.compile(
        loss="binary_crossentropy",
        # optimizer=optimizers.Adam(),
        optimizer=optimizers.RMSprop(lr=2e-5),
        metrics="binary_accuracy",
    )
    model.summary()
    return model

# model = build_model()

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_datagen.fit(x)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1.0 / 255)


callBackList = []
callBackList.append(callbacks.ModelCheckpoint("model/model.h5", save_best_only=True))
steps_per_epoch = int(trX.shape[0] / batch_size)

history = model.fit(train_datagen.flow(trX, trY, batch_size=batch_size), steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=2,
                             callbacks=callBackList, validation_data=test_datagen.flow(teX, teY, batch_size=batch_size), validation_steps=validation_steps)
