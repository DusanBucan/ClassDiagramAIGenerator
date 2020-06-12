import os
import cv2
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import img_to_array

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def extract_features_with_cnn():
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    path = '/dataset/300x300'
    train_images_class = []
    train_labels_class = []

    for e, i in enumerate(os.listdir(path)):
        try:
            if i.endswith(".jpg") and "klasa" in i:
                image = cv2.cvtColor(cv2.imread(os.path.join(path, i)), cv2.COLOR_BGR2RGB)
                filename = i.split(".")[0]
                df = pd.read_csv(os.path.join(path, filename + "_entires.groundtruth.txt"), header=None)
                gtvalues = []
                for row in df.values:
                    data = row[0].split(" ")
                    x1 = int(data[0])
                    y1 = int(data[1])
                    x2 = int(data[2])
                    y2 = int(data[3])
                    label = data[4].strip()
                    gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2, "label": label})

                if gtvalues[0]["label"] != "klasa":
                    continue

                ss.setBaseImage(image)
                ss.switchToSelectiveSearchQuality()
                ssresults = ss.process()
                imout = image.copy()
                postiveCounter = 0
                falseCounter = 0
                flag = 0
                fflag = 0
                bflag = 0

                for e, result in enumerate(ssresults):
                    if flag == 0:
                        for gtval in gtvalues:
                            ground_label = gtval["label"]
                            x, y, w, h = result

                            if w * h < 140 * 140:
                                continue

                            iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})

                            if iou > 0.90:
                                if postiveCounter < 15:
                                    timage = imout[y:y + h, x:x + w]

                                    resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                                    resized = img_to_array(resized)

                                    if ground_label == "klasa":
                                        train_images_class.append(resized)
                                        train_labels_class.append(ground_label)
                                    postiveCounter += 1
                                else:
                                    fflag = 1
                            if iou < 0.4:
                                if falseCounter < 8:
                                    timage = imout[y:y + h, x:x + w]
                                    resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                                    resized = img_to_array(resized)
                                    if ground_label == "klasa":
                                        train_images_class.append(resized)
                                        train_labels_class.append("background")
                                    falseCounter += 1
                                else:
                                    bflag = 1

                        if fflag == 1 and bflag == 1:
                            flag = 1

                # print("ucitao sliku: " + i)
                # print("positive: " + str(postiveCounter))
                # print("negative: " + str(falseCounter))
                # print("-----------------------")

        except Exception as e:
            print(e)
            print("error in " + filename)
            continue

    X_new_class = np.array(train_images_class, dtype=np.float) / 255.0
    y_new_class = np.array(train_labels_class)

    Y_class = np.array([1 if l == "klasa" else 0 for l in y_new_class])

    return X_new_class, Y_class


def load_svm():
    file = open('models/SVM_CNN', 'rb')
    return pickle.load(file)


def test(base_model):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    svm = load_svm()

    path = '/dataset/test/d11.jpg'
    image = cv2.imread(path)
    image = cv2.resize(image, (2048, 1152), interpolation=cv2.INTER_AREA)
    plt.imshow(image)
    plt.show()

    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()
    ssresults = ss.process()
    imout = image.copy()
    predictedWindows = []
    for e, result in enumerate(ssresults):
        if e < 3500:
            x, y, w, h = result

            if not (w * h > 40000 or (w > 300 and h > 80) or (h > 300 and w > 80)):
                continue

            timage = imout[y:y + h, x:x + w]
            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)

            to_predict = np.asarray([resized], dtype=np.float32) / 255.0

            features = base_model.predict(to_predict)

            # print(features.shape)

            features = features.reshape((features.shape[0], 512 * 7 * 7))

            # print(features.shape)

            scores = svm.predict_proba(features)[0]

            print(scores)
            print("------------")

            if scores[1] >= 0.97:
                plt.imshow(timage)
                plt.show()


if __name__ == '__main__':
    # X_new_class, Y_new_class = extract_features_with_cnn()
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_tensor=Input(shape=(224, 224, 3)),
                       input_shape=(224, 224, 3))
    # features = base_model.predict(X_new_class, batch_size=32, verbose=1)
    #
    # features = features.reshape((features.shape[0], 512 * 7 * 7))
    #
    # x_train, x_test, y_train, y_test = train_test_split(features, Y_new_class, test_size=0.2,
    #                                                     random_state=42)  # 0.2 posto je za trening
    # # print('Train shape: ', x_train.shape, y_train.shape)
    # # print('Test shape: ', x_test.shape, y_test.shape)
    #
    # clf_svm = SVC(kernel='linear', probability=True)
    # clf_svm.fit(x_train, y_train)
    # y_train_pred = clf_svm.predict(x_train)
    # y_test_pred = clf_svm.predict(x_test)
    # print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    # print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))
    #
    # file_svm = open('/models/SVM_CNN', 'wb')
    # pickle.dump(clf_svm, file_svm)
    #
    test(base_model)
