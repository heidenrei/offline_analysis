from tensorflow.keras.applications import EfficientNetB0 as Net
# from tensorflow.keras.applications import center_crop_and_resize, preprocess_input

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_labeler import prepare_for_training


batch_size = 48

width = 150
height = 150
epochs = 20
NUM_TRAIN = 2
NUM_TEST = 1
dropout_rate = 0.2
input_shape = (height, width, 3)
train_dir = os.path.join(os.curdir, 'train_dir')

# loading pretrained conv base model
conv_base = Net(weights="imagenet", include_top=False, input_shape=input_shape)

dropout_rate = 0.2
model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
# model.add(layers.Flatten(name="flatten"))
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
# model.add(layers.Dense(256, activation='relu', name="fc1"))
model.add(layers.Dense(2, activation="softmax", name="fc_out"))

print('This is the number of trainable layers '
      'before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights))

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

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

x, y = prepare_for_training(train_dir)

train_datagen.fit(x)


te_index = int(x.shape[0] * split)
teX, teY = x[0: te_index], y[0: te_index]
trX, trY = x[te_index:], y[te_index:]


train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    train_dir,
    # All images will be resized to target height and width.
    target_size=(height, width),
    batch_size=batch_size,
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode="categorical",
)

# validation_generator = test_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(height, width),
#     batch_size=batch_size,
#     class_mode="categorical",
# )

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=["acc"],
)

history = model.fit_generator(train_datagen.flow(trX, trY, batch_size=1), steps_per_epoch=2, epochs=10, verbose=2)
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=NUM_TRAIN // batch_size,
#     epochs=epochs,
#     # validation_data=validation_generator,
#     # validation_steps=NUM_TEST // batch_size,
#     verbose=2,
#     use_multiprocessing=True,
#     workers=2,
# )


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_x = range(len(acc))

plt.plot(epochs_x, acc, 'bo', label='Training acc')
plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs_x, loss, 'bo', label='Training loss')
plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# conv_base.trainable = True

# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name == 'multiply_16':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False