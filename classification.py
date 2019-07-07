import numpy as np
import sys
import random
import os
import tqdm
from PIL import ImageFile
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing import image
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler

from layer_rotation_control import Adam
from layer_rotation_monitoring import LayerRotationCurves


# TODO fix this constants
ImageFile.LOAD_TRUNCATED_IMAGES = True

img_width, img_height = 224, 224
CATEGORIES = range(16)
CHECKPOINT_DIR = './checkpoints/'
checkpoint_path = os.path.join(CHECKPOINT_DIR,
                               "weights-{epoch:02d}-{val_acc:.2f}.hdf5")

# TODO get this numbers automatically
resnet_weights = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
nb_validation_samples = 1147
nb_train_samples = 13650


def train(arch_file="model.json", weights_file='model.h5',
          train_data_dir="train", validation_data_dir="sample",
          arch='nasnet', epochs=500, batch_size=16,
          tb_callback=True, checkpoints=True):

    input_shape = (img_width, img_height, 3)
    shape = (img_width, img_height)

    # # Choose network to use
    if arch == 'resnet':
        from keras.applications.resnet50 import ResNet50
        pretrained_model = ResNet50(include_top=False,
                                    input_shape=input_shape,
                                    weights='imagenet')  # resnet_weights)
    if arch == 'nasnet':
        from keras.applications.nasnet import NASNetLarge
        pretrained_model = NASNetLarge(input_shape=None,
                                       include_top=True,
                                       weights='imagenet',
                                       input_tensor=None,
                                       pooling=None,
                                       classes=1000)

    if pretrained_model.output.shape.ndims > 2:
        output = Flatten()(pretrained_model.output)
    else:
        output = pretrained_model.output

    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(512, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(len(CATEGORIES), activation='softmax')(output)
    model = Model(pretrained_model.input, output)
    for layer in pretrained_model.layers:
        layer.trainable = False
    model.summary(line_length=200)
    lr = 0.001
    opt = Adam(lr, layca=True)

    # a keras callback to record layer rotation
    lrc = LayerRotationCurves()

    # learning rate schedule: divide learning rate by 5 at epochs 70 and 90
    def schedule(epoch):
        new_lr = lr
        if epoch > 70:
            new_lr *= 0.2
        if epoch > 90:
            new_lr *= 0.2
        return new_lr

    lrs = LearningRateScheduler(schedule)

    callbacks = [lrc, lrs]
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  weighted_metrics=['accuracy'],
                  metrics=['accuracy'])

    # Save model arch to json
    model_json = model.to_json()
    with open(arch_file, "w") as json_file:
        json_file.write(model_json)

    # Data augmentation with generators
    _train_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                    rotation_range=15,
                                    width_shift_range=.15,
                                    height_shift_range=.15,
                                    shear_range=0.15,
                                    zoom_range=0.15,
                                    channel_shift_range=1,
                                    horizontal_flip=True,
                                    vertical_flip=False)

    _test_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=15,
                                   width_shift_range=.15,
                                   height_shift_range=.15,
                                   shear_range=0.15,
                                   zoom_range=0.15,
                                   channel_shift_range=1,
                                   horizontal_flip=True,
                                   vertical_flip=False)

    train_generator = _train_gen.flow_from_directory(train_data_dir,
                                                     target_size=shape,
                                                     batch_size=batch_size,
                                                     class_mode='categorical')

    val_generator = _test_gen.flow_from_directory(validation_data_dir,
                                                  target_size=shape,
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

    if tb_callback:
        tb_callback = TensorBoard(log_dir='./tb_callback',
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=False)
        callbacks.append(tb_callback)

    if checkpoints:
        cp_callback = ModelCheckpoint(checkpoint_path,
                                      monitor='val_acc',
                                      save_weights_only=True,
                                      save_best_only=True,
                                      verbose=1)
        callbacks.append(cp_callback)
    model.fit_generator(train_generator,
                        steps_per_epoch=500,
                        validation_data=val_generator,
                        validation_steps=75,
                        epochs=epochs,
                        workers=4,
                        callbacks=callbacks)

    model.save_weights(weights_file)
    # plot layer rotation curves
    # plt.figure()
    # lrc.plot()


def preprocess_img(img_path):
    img_path = img_path
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


def test(arch_file='model.json', weights_file='model.h5', test_dir='sample',
         arch='nasnet'):  #  TODO ARCH DEFINED BY ARCH FILE
    with open(arch_file, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    if arch == 'nasnet':
        from keras.applications.nasnet import preprocess_input
    else:
        from keras.applications.resnet50 import preprocess_input
    loaded_model.load_weights(weights_file)
    y_true = []
    y_pred = []
    for _class in tqdm.tqdm(os.listdir(test_dir)):
        if _class == '.gitkeep':
            continue
        files = os.listdir(os.path.join(test_dir, _class))
        for _file in files:
            img = preprocess_img(os.path.join(test_dir, _class, _file))
            _predicted_class = np.argmax(loaded_model.predict(img))
            y_true.append(int(_class))
            y_pred.append(int(_predicted_class))
    report = classification_report(y_true, y_pred)
    score = balanced_accuracy_score(y_true, y_pred)
    print('Balanced score:', score)
    print(confusion_matrix(y_true, y_pred))
    return report


def get_random_sample_from_dataset(output_dir, dataset_dir='train'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for _class in os.listdir(dataset_dir):
        if not os.path.isdir(os.path.join(dataset_dir, _class)):
            continue
        files = os.listdir(os.path.join(dataset_dir, _class))
        dest_folder = os.path.join(output_dir, _class)
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        random.shuffle(files)
        files = files[:int(.2 * len(files))]
        for file in files:
            from_file = os.path.join(dataset_dir, _class, file)
            to_file = os.path.join(output_dir, _class, file)
            print("Adding {}".format(from_file))
            os.system('mv {} {}'.format(from_file, to_file))


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("BAD ARGS")
    if sys.argv[1] == 'train':
        print("TRAIN")
        gpu_devs = os.getenv('CUDA_VISIBLE_DEVICES', None)
        print(train())
    if sys.argv[1] == 'test':
        print("TEST")
        print(test(weights_file=sys.argv[2]))
    if sys.argv[1] == 'make_sample':
        print("MAKING NEW SAMPLE")
        get_random_sample_from_dataset(output_dir=sys.argv[2])
