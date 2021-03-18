from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

from tensorflow.keras.applications import EfficientNetB0 as Net
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import losses
from tensorflow.keras import metrics

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cv2

train_dir = '/home/gavin/offline_analysis/data/train/'
validation_dir = '/home/gavin/offline_analysis/data/valid/'
test_dir = '/home/gavin/offline_analysis/data/test/'

batch_size = 3

width = 224
height = 224
epochs = 150
NUM_TRAIN = len(os.listdir(train_dir + '1')) + len(os.listdir(train_dir + '0'))
NUM_TEST = len(os.listdir(test_dir + 'predict'))
NUM_VAL = len(os.listdir(validation_dir + '1')) + len(os.listdir(validation_dir + '0'))

print(NUM_TRAIN, NUM_TEST, NUM_VAL)

dropout_rate = 0.2
# dropout_rate = 0
input_shape = (height, width, 1)


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        directory=train_dir,
        color_mode="grayscale",
        # All images will be resized to target height and width.
        target_size=(height, width),
        batch_size=batch_size,
        shuffle=True,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='binary')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        directory=validation_dir,
        color_mode="grayscale",
        target_size=(height, width),
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)


def model1():
    conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape)

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D(name="gap"))
    # model.add(layers.Flatten(name="flatten"))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="dropout_out"))
    # model.add(layers.Dense(256, activation='relu', name="fc1"))
    # model.add(layers.Dense(2, activation='softmax', name="fc_out"))
    model.add(layers.Dense(2, activation='softmax', name="fc_out"))


    model.summary()

    conv_base.trainable = False


    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.RMSprop(lr=2e-1),
                metrics=['acc'])

    return model


def model2():
    efficient_net = Net(
        weights='imagenet',
        input_shape=(224,224,3),
        include_top=False,
        pooling='max'
    )

    model = models.Sequential()
    model.add(efficient_net)
    #model.add(layers.Dense(units = 120, activation='relu'))
    model.add(layers.Dropout(0.2, name="dropout_out"))
    model.add(layers.Dense(units = 64, activation = 'relu'))
    model.add(layers.Dense(units = 32, activation = 'relu'))
    model.add(layers.Dense(units = 16, activation = 'relu'))
    model.add(layers.Dense(units = 1, activation='sigmoid'))
    model.summary()

    efficient_net.trainable = False

    model.compile(optimizer=optimizers.Adam(lr=0.000001), loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model

def plot1(history):
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

def plot2(history):
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
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

if __name__ == '__main__':
    # model = model1()
    model = model2()
    history = model.fit(
        train_generator,
        #steps_per_epoch= NUM_TRAIN //batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        #validation_steps= NUM_VAL // batch_size,
        verbose=1,
        #use_multiprocessing=True,
        #workers=4
        )

    print(history.history.keys())

    plot2(history)
        
    pred = model.predict_generator(test_generator)
    print(pred)
    print(type(pred))

    # model.evaluate_generator(generator=validation_generator, steps=1, verbose=1)
    # test_generator.reset()
    # pred = model.predict_generator(test_generator, steps=1, verbose=1)
    # predicted_class_indices = np.argmax(pred, axis=1)
    # labels = (train_generator.class_indices)
    # labels = dict((v, k) for v, k in labels.items())
    # predictions = [labels[k] for k in predicted_class_indices]

    # filenames=test_generator.filenames
    # results=pd.DataFrame({"Filename":filenames, "Predictions":predictions})
    # results.to_csv("results.csv",index=False)
    # plot1(history)