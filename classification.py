from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.applications import ResNet50
from tensorflow.keras import backend as K
import cv2
import numpy as np
import sys
from PIL import ImageFile

# TODO fix this constants
ImageFile.LOAD_TRUNCATED_IMAGES = True
img_width, img_height = 640, 480
CATEGORIES = range(2)#range(16)
checkpoint_path = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
resnet_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# TODO get this numbers automatically
nb_validation_samples = 1147
nb_train_samples = 13650


def train(arch_file="model.json", weights_file='model.h5',
          train_data_dir="train", validation_data_dir="test",
          arch='own_model', epochs=10, batch_size=8,
          tb_callback=False, checkpoints=True):

    # Define shape
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    shape = (img_width, img_height)

    # Choose network to use
    if arch == 'own_model':
        model = Sequential()
        model.add(Conv2D(32, (2, 2), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(CATEGORIES)))
        model.add(Activation('softmax'))
    if arch == 'resnet':
        model = Sequential()
        model.add(ResNet50(include_top=False,
                           pooling='avg',
                           weights=resnet_weights_path))
        model.add(Dense(len(CATEGORIES), activation='softmax'))
        model.layers[0].trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Save model arch to json
    model_json = model.to_json()
    with open(arch_file, "w") as json_file:
        json_file.write(model_json)

    # Data augmentation with generators
    _train_gen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

    _test_gen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

    train_generator = _train_gen.flow_from_directory(train_data_dir,
                                                     target_size=shape,
                                                     batch_size=batch_size,
                                                     class_mode='categorical')

    val_generator = _test_gen.flow_from_directory(validation_data_dir,
                                                  target_size=shape,
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

    callbacks = []
    if tb_callback:
        tb_callback = TensorBoard(log_dir='./tb_callback',
                                  update_freq='batch',
                                  write_graph=True,
                                  write_images=True)
        callbacks.append(tb_callback)

    if checkpoints:
        cp_callback = ModelCheckpoint(checkpoint_path,
                                      monitor='val_acc',
                                      save_weights_only=True,
                                      save_best_only=True,
                                      verbose=1)
        callbacks.append(cp_callback)

    model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        validation_data=val_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        epochs=epochs,
                        callbacks=callbacks)

    model.save_weights(weights_file)


def preprocess_img(img_path):
    img = cv2.imread(img_path)
    img = img / 255.
    img = cv2.resize(img, (img_width, img_height))
    img = np.reshape(img, [1, img_width, img_height, 3])
    return img


def test(arch_file='model.json', weights_file='model.h5'):
    with open(arch_file, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_file)
    img = preprocess_img('test.jpg')
    _probs = loaded_model.predict_proba(img)
    _class = loaded_model.predict_classes(img)
    return _probs, _class


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("TRAIN")
        print(train())
    else:
        print("TEST")
        print(test())
