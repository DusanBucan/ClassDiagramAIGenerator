import os
import cv2
import pickle
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def load_svm_char():
    file_svm = open('models/SVM_chars', 'rb')
    return pickle.load(file_svm)


def resize_region_OCR(region):
    return cv2.resize(region, (32, 32), interpolation=cv2.INTER_NEAREST)


def load_images_OCR(test_images=False):
    # google drive
    path = '/content/drive/My Drive/cdSoft/dataset/znakovi'
    train_images = []
    train_labels = []
    type_images = "train"
    if test_images:
        type_images = "test"

    for e, i in enumerate(os.listdir(path)):
        for e1, j in enumerate(os.listdir(path + "/" + i)):
            if j == type_images:
                for e2, img in enumerate(os.listdir(path + "/" + i + "/" + j)):
                    if img.endswith(".jpg") or img.endswith(".png"):
                        image = cv2.imread(os.path.join(path, i, j, img))
                        img_words_bin = image

                        if img_words_bin.shape[0] != 32:
                            img_words_bin = resize_region_OCR(img_words_bin)

                        train_images.append(img_words_bin)
                        train_labels.append(i)

    return train_images, train_labels


def train_OCR_NN(base_model):
    train_images, train_labels = load_images_OCR()

    train_images = np.asarray(train_images)

    features = base_model.predict(train_images, batch_size=32, verbose=1)
    features = features.reshape((features.shape[0], 512 * 1 * 1))

    clf_svm = SVC(kernel='linear', probability=True)
    clf_svm.fit(features, train_labels)

    file_svm = open('/models/SVM_chars', 'wb')
    pickle.dump(clf_svm, file_svm)


def predict_char(char_image, svm):
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(32, 32, 3)),
                       input_shape=(32, 32, 3))

    char_image = np.asarray([char_image])
    features = base_model.predict(char_image, batch_size=32, verbose=1)
    features = features.reshape((features.shape[0], 512 * 1 * 1))
    return svm.predict(features)


def test_OCR_NN(base_model):
    svm = load_svm_char()
    test_images, ground_truth_labels = load_images_OCR(True)

    test_images = np.asarray(test_images)
    features = base_model.predict(test_images, batch_size=32, verbose=1)
    features = features.reshape((features.shape[0], 512 * 1 * 1))
    y_predicted_test = svm.predict(features)

    print(accuracy_score(ground_truth_labels, y_predicted_test))


if __name__ == '__main__':
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(32, 32, 3)),
                       input_shape=(32, 32, 3))
    train_OCR_NN(base_model)
    test_OCR_NN(base_model)
