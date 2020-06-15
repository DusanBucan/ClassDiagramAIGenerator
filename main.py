import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input

from train_char import load_svm_char, predict_char, resize_region_OCR
from train_class import load_svm
from train_relationship import load_svm_relationship
from generate_code import Class, add_relationship, make_project


def load_image(path):
    return cv2.imread(path)


def resize_image(image):
    return cv2.resize(image, (1024, 576), interpolation=cv2.INTER_NEAREST)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def image_bin_sobel(image_sobel, avg_value, max_value):
    height, width = image_sobel.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_sobel, avg_value, max_value, cv2.THRESH_BINARY)
    return image_bin


def dilate(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((2, 2))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=2)


def select_roi_class(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 100 or h > 100:
            region = image_orig[y:y + h + 1, x:x + w + 1]
            regions_array.append([region, (x, y, w, h)])
            # cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 5)

    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    # plt.imshow(image_orig)
    # plt.show()

    index = 0
    while index < len(regions_array):
        current = regions_array[index]
        for idx in range(index + 1, len(regions_array)):
            next_rect = regions_array[idx]

            result = 0
            x, y, w, h = current[1]
            x1, y1, w1, h1 = next_rect[1]
            height = max(h, h1, w, w1) * 0.5

            if y < y1 and y + h + height < y1:
                continue
            elif y > y1 and y1 + h1 + height < y:
                continue
            elif w < w1:
                result = line_matching(next_rect[1], current[1]) / w1
            else:
                result = line_matching(current[1], next_rect[1]) / w

            # spajanje dva region ako se njihova sirina preklapa 0.8 i dovoljno je blizu
            if result > 0.8:
                x2 = min(x, x1)
                y2 = min(y, y1)
                w2 = max(x + w, x1 + w1) - x2
                h2 = max(y + h, y1 + h1) - y2
                region = image_orig[y2:y2 + h2 + 1, x2:x2 + w2 + 1]
                regions_array[index] = [region, (x2, y2, w2, h2)]
                del regions_array[idx]
                break
        else:
            index += 1

    # da se sklone horizontalne veze koje su upale
    regions = []
    for region in regions_array:
        x, y, w, h = region[1]
        if w * 3 > h and h * 3 > w:
            regions.append(region)
    return regions


def line_matching(bigger, smaller):
    if bigger[0] < smaller[0] and bigger[0] + bigger[2] > smaller[0] + smaller[2]:
        return smaller[2]
    elif bigger[0] < smaller[0]:
        return bigger[0] + bigger[2] - smaller[0]
    elif bigger[0] + bigger[2] > smaller[0] + smaller[2]:
        return smaller[0] + smaller[2] - bigger[0]


def performSobel(image, direction="horizontal", line_size=31):
    img_gs = image_gray(image)
    if direction == "horizontal":
        sobelx64f = cv2.Sobel(img_gs, cv2.CV_64F, 0, 1, ksize=line_size)
    else:
        sobelx64f = cv2.Sobel(img_gs, cv2.CV_64F, 1, 0, ksize=line_size)

    sobelx64f = np.abs(sobelx64f)

    sobelx64f = dilate(sobelx64f)
    sobelx64f = dilate(sobelx64f)
    sobelx64f = erode(sobelx64f)

    sobelx64f = erode(sobelx64f)
    # sobelx64f = erode(sobelx64f)
    # make binary images from sobel transformed images
    sobelx64f_bin = image_bin_sobel(sobelx64f, np.average(sobelx64f) * 2.4, np.max(sobelx64f))

    sobelx64f_bin = sobelx64f_bin / np.max(sobelx64f_bin)
    sobelxUint8 = np.asarray(sobelx64f_bin, dtype=np.uint8)

    plt.imshow(sobelxUint8, 'gray')
    plt.show()
    return sobelxUint8


def findRelationShipsRegions(resized_image, direction="horizontal"):
    sobelxUint8 = performSobel(resized_image, direction)
    regions = select_roi_class(resized_image, sobelxUint8)

    # boje da bude recnik kasnije da uzimas sta ti treba...
    # for region in sorted_regions:
    #   region.append(direction)

    return regions


def resize_region_cnn(region):
    height, width, depth = region.shape
    max_dim = max(height, width)
    max_dim_img = np.zeros([max_dim, max_dim, 3], dtype=np.uint)
    max_dim_img.fill(255)
    # preslikas u gornji levi ugao celu ulaznu sliku, ostalo je crno
    # da li ce to crno da utice na to da on predvidi klasu? a jbg nzm..
    # nisam ga tako trenirao nego onako da iseze i to sto je isekao
    # da resize iz mozda mora ovako da se istrenirao
    for h in range(0, len(region)):
        for w in range(0, len(region[h])):
            max_dim_img[h][w] = region[h][w]
    return cv2.resize(max_dim_img, (224, 224), interpolation=cv2.INTER_NEAREST)


def find_relationships(resized_image, class_array):
    model_rs = load_svm_relationship()
    relationship_dic = {}

    base_model_relationship = VGG16(weights='imagenet', include_top=False,
                                    input_tensor=Input(shape=(300, 300, 3)),
                                    input_shape=(300, 300, 3))

    for idx in range(0, len(class_array) - 1):
        x, y, w, h = class_array[idx].img[1]
        for i in range(idx + 1, len(class_array)):
            rot = False
            x1, y1, w1, h1 = class_array[i].img[1]
            if abs(x - x1) > abs(y - y1) and (y + h < y1 or y1 + h1 < y):
                continue
            elif abs(x - x1) <= abs(y - y1) and (x + w < x1 or x1 + w1 < x):
                continue
            elif abs(x - x1) > abs(y - y1):
                y2 = min(y, y1)
                h2 = max(y + h, y1 + h1) - y2
                if x < x1:
                    x2 = int(x + w * 0.8)
                    w2 = int(x1 + w1 * 0.2 - x2)
                else:
                    x2 = int(x1 + w1 * 0.8)
                    w2 = int(x + w * 0.2 - x2)
            else:
                x2 = min(x, x1)
                w2 = max(x + w, x1 + w1) - x2
                if y < y1:
                    y2 = int(y + h * 0.8)
                    h2 = int(y1 + h1 * 0.2 - y2)
                else:
                    y2 = int(y1 + h1 * 0.8)
                    h2 = int(y + h * 0.2 - y2)
                rot = True

            region = resized_image[y2:y2 + h2 + 1, x2:x2 + w2 + 1]
            resized_region = cv2.resize(region, (300, 300), interpolation=cv2.INTER_NEAREST)

            if rot:
                (h, w) = resized_region.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, 90, 1.0)
                resized_region = cv2.warpAffine(resized_region, M, (h, w))

            plt.imshow(resized_region)
            plt.show()

            a = np.asarray([resized_region])

            features = base_model_relationship.predict(a, batch_size=32, verbose=1)
            features = features.reshape((features.shape[0], 512 * 9 * 9))
            scores = model_rs.predict(features)
            print("scores: ", scores)
            add_relationship(scores, class_array[idx], class_array[i])
            # max_score = np.max(scores[0])
            # max_score_indx = np.argmax(scores[0])
            # print(max_score_indx)
            # print(scores)



def calculate_row_distances(row):
    row_distances = []
    for m in range(len(row) - 1):
        firstContoure = row[m]
        secondContoure = row[m + 1]

        x1, y1, w1, h1 = firstContoure[1]
        x2, y2, w2, h2 = secondContoure[1]

        distance = x2 - (x1 + w1)
        row_distances.append(distance)

    return row_distances


def extract_rows_OCR(class_image):
    rows = []
    distances = []

    class_image_gray = image_gray(class_image)
    edges = cv2.Canny(class_image_gray, 120, 200)
    plt.imshow(edges, cmap='gray')
    plt.show()

    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 7 and w < 50 and h > 7 and h < 60:
            region = class_image[y:y + h + 1, x:x + w + 1]
            regions_array.append([region, (x, y, w, h)])
    regions_array = sorted(regions_array, key=lambda x: x[1][1] + x[1][3])

    j = 0
    i = 0
    while i < len(regions_array):
        row = []
        # koristi da se odredi sa K-means sta je razlika izmedju slova, a sta izmedju reci u jednom redu
        row_distances = []
        firstRowRect = regions_array[i]
        y_row = firstRowRect[1][1] + firstRowRect[1][3]

        for character in regions_array[i:]:
            character_bottom_y = character[1][1] + character[1][3]
            j += 1
            if abs(character_bottom_y - y_row) < 30:
                row.append(character)
            else:
                #  od tog pocinje novi red
                break

        # izdvojen je ceo jedan red
        row = sorted(row, key=lambda x: x[1][0])
        # da preskocis sve konture koje su u tom redu da njih ne gleda petlja
        i = j
        # izracunaj distance u svakom row izmedju kontura.
        row_distances = calculate_row_distances(row)
        distances += row_distances
        rows.append(row)

    distances = np.asarray(distances).reshape(-1, 1)
    k_means = KMeans(n_clusters=2, random_state=0).fit(distances)
    return rows, k_means, regions_array


def process_row_OCR(row, row_indx, kmeans, svm_char):
    words = []
    word = ""
    distances = calculate_row_distances(row)
    # ovo je da posle poslednje konture u redu nista ne znaci
    # koje je rastojanje a da ne moram da petljam sa indeksima
    distances.append(-100)
    distance_types = kmeans.predict(np.asarray(distances).reshape(-1, 1))

    for region_indx, row_region in enumerate(row):

        row_region[0] = resize_region_OCR(row_region[0])
        char_string = predict_char(row_region[0], svm_char)
        word += char_string[0]

        dist_type = distance_types[region_indx]
        # da li ce rastojanje izmedju reci uvek da bude 1
        # a rastojanje unutar reci da bude 0 u kmeans..
        if dist_type == 1:
            words.append(word)
            word = ""

    if word not in words:
        words.append(word)

    return words

def perform_class_OCR(OCR_region, index):
    class_image = cv2.resize(OCR_region[0], (300, 300), interpolation=cv2.INTER_AREA)
    char_svm = load_svm_char()



    rows, kmeans, all_regions = extract_rows_OCR(class_image)
    for indx, row in enumerate(rows):
        row_words = process_row_OCR(row, indx, kmeans, char_svm)
        print("slika: ", index, " index reda: ", indx,  " procitao: ", row_words)


    for reg in all_regions:
        (x, y, w, h) = reg[1]
        cv2.rectangle(class_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(class_image)
    plt.show()

    generated_class = Class("Klasa" + str(index), OCR_region)
    return generated_class


if __name__ == '__main__':
    img = load_image('dataset/test/d11.jpg')
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_tensor=Input(shape=(224, 224, 3)),
                       input_shape=(224, 224, 3))
    svm = load_svm()
    print(len(img), len(img[0]))

    resized_image = resize_image(img)
    regions_horizontal = findRelationShipsRegions(resized_image, "horizontal")

    # mozes da prodjes kroz sve njih i da ih guras kroz mrezu, one koje klasifikuje kao
    # kao uzmes i dodas ih u recnik..
    # posle proveris da li su povezani sa nekim klasama koje su u recniku..
    # ako nisu onda ih izbacis. ---> ostaje ti da odredis smer veze.. i da poboljsas mrezu
    # kad klasifikuje veze.

    class_array = []
    n = 1
    for region in regions_horizontal:
        resized = cv2.resize(region[0], (224, 224), interpolation=cv2.INTER_AREA)
        to_predict = np.asarray([resized], dtype=np.float32) / 255.0

        features = base_model.predict(to_predict)
        features = features.reshape((features.shape[0], 512 * 7 * 7))
        scores = svm.predict_proba(features)[0]

        if scores[1] >= scores[0]:
            c = perform_class_OCR(region, n)
            class_array.append(c)
            n += 1

    find_relationships(resized_image, class_array)

    print("******************************")
    for img in class_array:
        print(img.name)
        for i in img.relationships:
            print(i.type)
        # print(img.relationships)
    print("******************************")

    make_project("./generated", "projekat", class_array)
