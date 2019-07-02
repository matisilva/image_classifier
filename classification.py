import cv2
import numpy as np
import sys
import random
import os
import tqdm
from PIL import ImageFile
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.applications import ResNet50
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# TODO fix this constants
ImageFile.LOAD_TRUNCATED_IMAGES = True

img_width, img_height = 224, 224
CATEGORIES = range(16)
CHECKPOINT_DIR = './checkpoints/'
checkpoint_path = os.path.join(CHECKPOINT_DIR,
                               "weights-{epoch:02d}-{val_acc:.2f}.hdf5")

# TODO get this numbers automatically
nb_validation_samples = 1147
nb_train_samples = 13650


def train(arch_file="model.json", weights_file='model.h5',
          train_data_dir="train", validation_data_dir="test",
          arch='resnet', epochs=100, batch_size=16,
          tb_callback=True, checkpoints=True):

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
                           weights='imagenet'))
        model.add(Dense(len(CATEGORIES), activation='softmax'))
        # model.layers[0].trainable = False
        for layer in model.layers[:-1]:
            layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  weighted_metrics=['accuracy'],
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
                                  # update_freq='batch',
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
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # #img = cv2.imread(img_path)
    # img = img / 255.
    # img = cv2.resize(img, (img_width, img_height))
    # if K.image_data_format() == 'channels_first':
    #     img = np.reshape(img, [1, 3, img_width, img_height])
    # else:
    #     img = np.reshape(img, [1, img_width, img_height, 3])
    # return img


def test(arch_file='model.json', weights_file='model.h5', test_dir='test'):
    with open(arch_file, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_file)
    y_true = []
    y_pred = []
    for _class in tqdm.tqdm(os.listdir(test_dir)):
        if _class == '.gitkeep':
            continue
        files = os.listdir(os.path.join(test_dir, _class))
        for _file in files:
            img = preprocess_img(os.path.join(test_dir, _class, _file))
            _predicted_class = loaded_model.predict_classes(img)
            y_true.append(int(_class))
            y_pred.append(_predicted_class[0])
    report = classification_report(y_true, y_pred)
    return report


def get_random_sample_from_dataset(dataset_dir='train', output_dir='sample'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for _class in os.listdir(dataset_dir):
        if not os.path.isdir(os.path.join(dataset_dir, _class)):
            continue
        files = os.listdir(os.path.join(dataset_dir, _class))
        dest_folder = os.path.join(output_dir, _class)
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        for _ in range(10):
            from_file = os.path.join(dataset_dir, _class, random.choice(files))
            to_file = os.path.join(output_dir, _class, random.choice(files))
            print("Adding {}".format(from_file))
            os.system('cp {} {}'.format(from_file, to_file))


# def test_santi(arch_file='model.json', weights_file='model.h5'):
#     with open(arch_file, 'r') as json_file:
#         loaded_model_json = json_file.read()
#     loaded_model = model_from_json(loaded_model_json)
#     loaded_model.load_weights(weights_file)
#     y_true = []
#     y_pred = []
#     for _class in tqdm.tqdm(os.listdir('test')):
#         if _class == '.gitkeep':
#             continue
#         files = os.listdir(os.path.join('test', _class))
#         random.shuffle(files)
#         files = files[:10]
#         for file_ in files:
#             img = preprocess_img(os.path.join('test', _class, file_))
#             _predicted_class = loaded_model.predict_classes(img)
#             y_true.append(int(_class))
#             y_pred.append(_predicted_class[0])
#     report = classification_report(y_true, y_pred)
#     return report


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("BAD ARGS")
    if sys.argv[1] == 'train':
        print("TRAIN")
        gpu_devs = os.getenv('CUDA_VISIBLE_DEVICES', None)
        if gpu_devs:
            import tensorflow as tf
            with tf.device('/device:XLA_GPU:0'):
                print(train())
        else:
            print(train())
    if sys.argv[1] == 'test':
        print("TEST")
        print(test(weights_file=sys.argv[2]))
    if sys.argv[1] == 'make_sample':
        print("MAKING NEW SAMPLE")
        get_random_sample_from_dataset()
