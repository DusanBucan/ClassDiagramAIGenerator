import time

from tensorflow.keras.models import load_model

from trainCharacterMaps import load_characters_test
from sklearn.preprocessing import LabelBinarizer
import numpy as np

def test_handWrittenCharacters_onlyWarmUp(test_data, labels):

    start_time = time.time()

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

    predIdxs = np.argmax(predIdxs, axis=1)  # ovo treba za svaku da vrati koju klasu je prediktovao...


    for i, idx in enumerate(predIdxs):
        label = lb.classes_[idx]
        true_label = labels[i]
        if label == true_label:
            correct += 1

    print(correct / len(predIdxs))
    print(time.time() - start_time)


if __name__ == '__main__':
    testData, testLabels, number_of_images = load_characters_test()
    test_handWrittenCharacters_onlyWarmUp(testData, testLabels)
