import os
import cv2
import pickle
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import img_to_array

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def load_data(purpose="train", rs_path="dataset/relationships/"):
    mapa = {}
    print(os.listdir(rs_path))
    for f in os.listdir(rs_path):
        print(f)
        for f2 in os.listdir(rs_path + f):
            if f2 == purpose:
                for img_name in os.listdir(rs_path + f + '/' + f2):
                    img_path = os.path.join(rs_path + f, f2, img_name)
                    image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
                    image = img_to_array(image)
                    if f not in mapa:
                        mapa[f] = [image]
                    else:
                        mapa[f].append(image)

    data = []
    labels = []
    for idx, key in enumerate(mapa.keys()):
        for j in mapa[key]:
            labels.append(key)
            data.append(j)

    data = np.array(data, dtype="float") / 255.0
    return data, labels


def load_svm_relationship():
    file = open('models/SVM_relationships', 'rb')
    return pickle.load(file)


if __name__ == '__main__':
    trainX, trainY = load_data()

    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(300, 300, 3)),
                       input_shape=(300, 300, 3))

    features = base_model.predict(trainX, batch_size=32, verbose=1)
    features = features.reshape((features.shape[0], 512 * 9 * 9))

    clf_svm = SVC(kernel='linear', probability=True)
    clf_svm.fit(features, trainY)

    testData, testLabels = load_data("test")
    features = base_model.predict(testData, batch_size=32, verbose=1)
    features = features.reshape((features.shape[0], 512 * 9 * 9))

    y_train_pred = clf_svm.predict(features)
    print(accuracy_score(testLabels, y_train_pred))

    file_svm = open('/models/SVM_relationships', 'wb')
    pickle.dump(clf_svm, file_svm)