import numpy as np
import cv2  # OpenCV
import matplotlib
import matplotlib.pyplot as plt
import collections
import numpy as np
import os
import csv

import keras
import tensorflow as tf

# # iscrtavanje slika u notebook-u
# %matplotlib inline
# # prikaz vecih slika
# matplotlib.rcParams['figure.figsize'] = 16,12


from tensorflow.keras.layers import *

# keras
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers.normalization import BatchNormalization
# from tensorflow.keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
# from keras.layers.core import Activation
# from keras.layers.core import Flatten
# from keras.layers.core import Dropout
# from keras.layers.core import Dense
# from keras import backend as K
#
# from keras.optimizers import SGD


from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
# from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


# from keras.models import load_model


# from keras.applications.vgg16 import VGG16
# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
# from keras.models import Model
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras.layers.core import Flatten
# from keras.engine.input_layer import Input


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import  preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Activation, Dropout, MaxPooling2D
from tensorflow.keras.layers import Input   # nisam siguran da li je dobar...
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

# from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# CONSTANTS
from resizeImage import resizeImage


CHARACTER_CNN_OUTPUTS = 68

EPOCHS_VGG16_INIT = 30  # moze da ide od 10-30 epoha dobro sam ubo.
EPOCHS_VGG16_FINETUNE = 30

EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (45, 45, 3) # ovih dimenzija su slova..

# initialize the data and labels
data = []
labels = []
data_dictionary = {}
train_dir = "data/train/"

test_data = []
test_labels = []
test_dictionary = {}
test_dir = "data/test/"


base_model = VGG16(weights='imagenet', include_top=False,
                       input_tensor=Input(shape=(IMAGE_DIMS[1], IMAGE_DIMS[0], 3)),
                    input_shape=(IMAGE_DIMS[1], IMAGE_DIMS[0], 3))


aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")


first_dir = 'podaci/Img/'
second_dir = 'podaci/s/'



def load_characters(purpose="train"):
    mapa = {}

    for folder in os.listdir(second_dir):
        if os.path.isdir(second_dir + folder):
            # ovo je citav folder za taj znak....
            for folder2 in os.listdir(second_dir + folder):
                for img_name in os.listdir(second_dir + folder + "/" + folder2):
                    img_path = os.path.join(second_dir + folder, folder2, img_name)
                    image = cv2.imread(img_path)
                    image = img_to_array(image)
                    if folder2 == purpose:
                        if folder not in mapa:
                            mapa[folder] = [image]
                        else:
                            mapa[folder].append(image)

    for folder in os.listdir(first_dir):
        if os.path.isdir(first_dir + folder):
            for folder2 in os.listdir(first_dir + folder):
                for img_name in os.listdir(first_dir + folder + "/" + folder2):
                    img_path = os.path.join(first_dir + folder, folder2, img_name)
                    image = cv2.imread(img_path)
                    image = resizeImage(image, IMAGE_DIMS)
                    image = img_to_array(image)
                    if folder2 == purpose:
                        if folder not in mapa:
                            mapa[folder] = [image]
                        else:
                            mapa[folder].append(image)

    # treba da
    data = []
    labels = []
    for idx, character in enumerate(mapa.keys()):
        for image in mapa[character]:
            labels.append(character)  # ovde sam stavljala indeks jer mi nije prihvatila kad sam koristila neku mrezu od pre
            image  = image / 255
            data.append(image)

    # sad imam oba vektora u data su slike u labeles su koje slovo treba da bude...
    labels = np.array(labels)

    # binarize the labels
    lb = LabelBinarizer()
    labels_coded = lb.fit_transform(labels)

    #svaki element od data je lista
    data = np.asarray(data, dtype=np.float32)


    # # partition the data into training and testing splits using 80% of
    # # the data for training and the remaining 20% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels_coded, test_size=0.2, random_state=42)

    return trainX, testX, trainY, testY, labels


def create_cnn_for_handWrittenCharacters(output_classes):

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    # and a logistic layer -- let's say we have 5 classes cveca
    predictions = Dense(output_classes, activation='softmax')(x)
    # # this is the model we will train
    modelVGG16 = Model(inputs=base_model.input, outputs=predictions)


    print(modelVGG16.summary())
    for (i, layer) in enumerate(modelVGG16.layers):
        print("[INFO] {}\t{}".format(i, layer.__class__.__name__))

    return modelVGG16


def init_weights_on_fc_for_handWrittenCharacters(model, trainX, trainY, testX, testY):
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    print("krece trening")


    H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS_VGG16_INIT, verbose=1)
    model.save('warmUpOnly54SizeVGG16_GPU.h5')



def fineTune_handWrittenCharacters():
    model = load_model('fineTune96SizeVGG16_GPU.h5')
    trainX, testX, trainY, testY, labels, distinct_labels = fineTune_load_characters()

    print("ucitao slike")

    for layer in model.layers[:15]:
        layer.trainable = False
    for layer in model.layers[15:]:
        layer.trainable = True

    # # we need to recompile the model for these modifications to take effect
    # # we use SGD with a low learning rate

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS_VGG16_FINETUNE, verbose=1)

    model.save('fineTune96SizeVGG16_GPU.h5')



def test_handWrittenCharacters_onlyWarmUp(test_data, labels):

    correct = 0

    # binarize the labels
    lb = LabelBinarizer()
    labels_test = lb.fit_transform(labels)

    model = load_model('fineTune96SizeVGG16_GPU.h5')

    predIdxs = model.predict(test_data)

    # dobijes za svaku sliku koja je unutar test_data po jednu listu
    # u toj listi imas za svih 62 klase verovatnoce..
    # uzmes najvecu
    # ovo moze da se koristi sa slidinig window-om i piramidama
    # da se prepoznaju slova gde su u klasi i ako je slovo u tom
    # sliding window-u onda da se I KOJE JE SLOVO...

    print(predIdxs[0])
    predIdxs = np.argmax(predIdxs, axis=1)  # ovo treba za svaku da vrati koju klasu je prediktovao...


    for i, idx in enumerate(predIdxs):
        label = lb.classes_[idx]
        true_label = labels[i]
        if label == true_label:
            correct += 1

    print(correct / len(predIdxs))

if __name__ == '__main__':
    # from __future__ import absolute_import, division, print_function, unicode_literals
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    trainX, testX, trainY, testY, labels = load_characters("train")
    print(len(trainX))
    print(len(testX))
    print(len(labels))
    #
    charcterCNN = create_cnn_for_handWrittenCharacters(CHARACTER_CNN_OUTPUTS)
    #
    init_weights_on_fc_for_handWrittenCharacters(charcterCNN, trainX, trainY, testX, testY)

    # testData, testLabels, number_of_images = load_characters_test()
    # print(number_of_images)

    # test_handWrittenCharacters_onlyWarmUp(testData, testLabels)

    # fineTune_handWrittenCharacters()
