import numpy as np
import cv2  # OpenCV
import matplotlib
import matplotlib.pyplot as plt
import collections
import numpy as np
import os
import csv

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


OUTPUT_CLASSES = 7
EPOCHS_ResNet152V2_INIT = 30  # moze da ide od 10-30 epoha dobro sam ubo.
EPOCHS_ResNet152V2_FINETUNE = 30

EPOCHS = 100
INIT_LR = 1e-3
BS = 16
IMAGE_DIMS = (224, 224, 3) # ovih dimenzija su slova..

# initialize the data and labels
data = []
labels = []
data_dictionary = {}

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")


first_dir = 'podaci/klase/'
second_dir = 'podaci/veze/'

al = ['klasa', 'zavisnost', 'generalizacija', 'realizacija', 'asocijacija', 'kompozicija', 'agregacija']



# za svaku vezu ucita po 50 slika a ostale ce biti za testiranje....
# za klase ucita 150 slika ostale ce biti za testiranje.

def load_train_data():
    mapa = {}
    for folder in os.listdir(second_dir):
        if os.path.isdir(second_dir + folder):
            j = 0
            # ovo je citav folder za taj tip veza
            for img_name in os.listdir(second_dir + folder):
                img_path = os.path.join(second_dir + folder, img_name)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
                image = img_to_array(image)
                if folder not in mapa:
                    mapa[folder] = [image]
                else:
                    mapa[folder].append(image)
                j += 1
                if j >= 50:
                    print("doslo do break ucitavanje veza")
                    break

    j = 0  # da izbrojimo 150 slika
    for img_name in os.listdir(first_dir):
        img_path = os.path.join(first_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        if 'klasa' not in mapa:
            mapa['klasa'] = [image]
        else:
            mapa['klasa'].append(image)
        j += 1
        if j >= 150:
            print("doslo do break ucitavanje klasa")
            break

    data = []
    labels = []
    for idx, i in enumerate(mapa.keys()):
        for j in mapa[i]:
            labels.append(i)  # ovde sam stavljala indeks jer mi nije prihvatila kad sam koristila neku mrezu od pre
            data.append(j)

    # sad imam oba vektora u data su slike u labeles su koje slovo treba da bude...
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # binarize the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    # # partition the data into training and testing splits using 80% of
    # # the data for training and the remaining 20% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
    return trainX, testX, trainY, testY, labels, len(al)



def create_cnn_for_Class_and_Relationships(output_classes):

    base_model = VGG16(weights='imagenet', include_top=False,
        input_tensor=Input(shape=(IMAGE_DIMS[1], IMAGE_DIMS[0], 3)),
        input_shape=(IMAGE_DIMS[1], IMAGE_DIMS[0], 3))

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


def warmUp_cnn_for_Class_and_Relationships(modelToWarmUp, trainX, trainY, testX, testY):

    for (j, modeLayer) in enumerate(modelToWarmUp.layers):
        if j < 19:
            modeLayer.trainable = False
        else:
            modeLayer.trainable = True

    modelToWarmUp.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    print("krece trening")

    H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS_ResNet152V2_INIT, verbose=1)

    model.save('warmUpOnly224SizeVGG16_GPU.h5')


def test_classDiagram_onlyWarmUp(test_data, labels):

    correct = 0

    # binarize the labels
    lb = LabelBinarizer()
    labels_test = lb.fit_transform(labels)

    model = load_model('warmUpOnly224SizeVGG16_GPU.h5')

    predIdxs = model.predict(test_data)

    # dobijes za svaku sliku koja je unutar test_data po jednu listu
    # u toj listi imas za svih 62 klase verovatnoce..
    # uzmes najvecu
    # ovo moze da se koristi sa slidinig window-om i piramidama
    # da se prepoznaju slova gde su u klasi i ako je slovo u tom
    # sliding window-u onda da se I KOJE JE SLOVO...

    predIdxs = np.argmax(predIdxs, axis=1)  # ovo treba za svaku da vrati koju klasu je prediktovao...


    for i, idx in enumerate(predIdxs):
        label = lb.classes_[idx]
        true_label = labels[i]
        if label == true_label:
            correct += 1

    print(correct / len(predIdxs))

def load_classDiagram_test():
    mapa = {}
    for folder in os.listdir(second_dir):
        if os.path.isdir(second_dir + folder):
            j = 0
            # ovo je citav folder za taj tip veza
            for img_name in os.listdir(second_dir + folder):
                if j >= 120:
                    img_path = os.path.join(second_dir + folder, img_name)
                    image = cv2.imread(img_path)
                    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
                    image = img_to_array(image)
                    if folder not in mapa:
                        mapa[folder] = [image]
                    else:
                        mapa[folder].append(image)
                j += 1
                if j >= 130:
                    print("doslo do break ucitavanje test veza")
                    break

    j = 0  # da izbrojimo 150 slika
    for img_name in os.listdir(first_dir):
        if j >= 161:
            img_path = os.path.join(first_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
            image = img_to_array(image)
            if 'klasa' not in mapa:
                mapa['klasa'] = [image]
            else:
                mapa['klasa'].append(image)
        j += 1
        if j >= 171:
            print("doslo do break ucitavanje klasa")
            break

    data = []
    labels = []
    for idx, i in enumerate(mapa.keys()):
        for j in mapa[i]:
            labels.append(i)  # ovde sam stavljala indeks jer mi nije prihvatila kad sam koristila neku mrezu od pre
            data.append(j)

    # sad imam oba vektora u data su slike u labeles su koje slovo treba da bude...
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    return data, labels




if __name__ == '__main__':

    # model = create_cnn_for_Class_and_Relationships(OUTPUT_CLASSES)
    # trainX, testX, trainY, testY, labels, classes = load_train_data()
    #
    # print(len(trainX))
    # print(len(testX))

    # warmUp_cnn_for_Class_and_Relationships(model, trainX, trainY, testX, testY)

    testData, testLabels = load_classDiagram_test()
    print(testData[0])
    # test_classDiagram_onlyWarmUp(testData, testLabels)
