from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing

IMG_SIZE = 528
# batch size should probably be smaller...
batch_size = 64
num_classes = 2


class Detector:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = self.build_model()
        print(self.model.summary())

    def augmentation_model(self):
        img_augmentation = keras.models.Sequential([
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),],
        name="img_augmentation",)
        
        return img_augmentation

    def build_model(self):
        img_augmentation = self.augmentation_model()
        inputs = keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

        x = img_augmentation(inputs)
        # deafult drop_connect_rate is 0.2
        model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet", drop_connect_rate=0.3)

        # Freeze the pretrained weights
        model.trainable = False

        # Rebuild top
        x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = keras.layers.BatchNormalization()(x)

        top_dropout_rate = 0.2
        x = keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = keras.layers.Dense(self.num_classes, activation="softmax", name="pred")(x)

        # Compile
        model = keras.Model(inputs, outputs, name="EfficientNet")
        optimizer = keras.optimizers.Adam(learning_rate=1e-2)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model


d = Detector(num_classes)