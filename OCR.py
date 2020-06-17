import cv2
import matplotlib.pyplot as plt
import numpy as np

from generate_code import Class
from train_char import load_svm_char, predict_char, resize_region_OCR
from sklearn.cluster import KMeans
from tss import read_char


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def calculate_row_distances(row):
    row_distances = []
    for m in range(len(row) - 1):
        first_contoure = row[m]
        second_contoure = row[m + 1]
        distance = second_contoure["x1"] - first_contoure["x2"]
        row_distances.append(distance)

    return row_distances


""" METODA DA IZDVOJI SLOVA KAO REGIONE sa SLIKE sa vezbi i ona sto smo je pre koristilis

    ULAZ: Slika vecline 300*300
    IZLAZ: regioni koji predstavlju slova
"""


def make_OCR_rectangles_old_roi(class_image):
    class_image_gray = image_gray(class_image)
    edges = cv2.Canny(class_image_gray, 120, 200)
    plt.imshow(edges, cmap='gray')
    plt.show()

    regions_array = []
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 7 and w < 50 and h > 7 and h < 60:
            region_image = class_image[y:y + h + 1, x:x + w + 1]
            region = {"x1": x, "x2": x + w, "y1": y, "y2": y + h, "image": region_image}
            regions_array.append(region)

    return regions_array


""" METODA DA IZDVOJI SLOVA KAO REGIONE sa SLIKE 

    ULAZ: Slika vecline 300*300
    IZLAZ: regioni koji predstavlju slova
"""


def make_OCR_rectangles(class_image):
    rectangles = []

    mser = cv2.MSER_create()
    gray = cv2.cvtColor(class_image, cv2.COLOR_BGR2GRAY)  # Converting to GrayScale

    plt.imshow(gray, 'gray')
    plt.show()
    # gray_img = class_image.copy()

    regions, _ = mser.detectRegions(gray)

    # IZ PRIMERA DA NE PRAVI PRAVUGAONIKE NEGO DA BAS IZDOVIJI SLOVO...
    # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    # cv2.polylines(gray_img, hulls, 1, (0, 0, 255), 2)
    for r in regions:
        x_cords = [a[0] for a in r]
        y_cords = [a[1] for a in r]
        x_min = min(x_cords)
        x_max = max(x_cords)
        y_min = min(y_cords)
        y_max = max(y_cords)
        w = x_max - x_min
        h = y_max - y_min

        if w > 60 or h > 50 or w * h > 2500 or (h < 5 and w > 30) or h * 10 < w or h > w * 10:
            continue

        rect = dict()
        rect["x1"] = x_min
        rect["x2"] = x_max
        rect["y1"] = y_min
        rect["y2"] = y_max

        rectangles.append(rect)

    nms_rectangles = []
    previous_len = -1
    iter = 0
    while True:
        if previous_len == len(nms_rectangles):
            break
        # if iter == 1:
        #     break
        else:
            if iter != 0:
                previous_len = len(nms_rectangles)
                rectangles = nms_rectangles
                nms_rectangles = []

        iter += 1

        rectangles = sorted(rectangles, key=lambda x: abs(x["x2"] - x["x1"]) * abs(x["y2"] - x["y1"]))
        while len(rectangles) > 0:
            similiar = []
            current_rect = rectangles[0]
            for rect in rectangles:
                inside = is_inside(current_rect, rect)
                if inside:
                    similiar.append(rect)
                else:
                    iou = get_iou(current_rect, rect)
                    if iou > 0.7:
                        similiar.append(rect)

            x_min = min([r["x1"] for r in similiar])
            x_max = max([r["x2"] for r in similiar])
            y_min = min([r["y1"] for r in similiar])
            y_max = max([r["y2"] for r in similiar])

            rect_nms = dict()
            rect_nms["x1"] = x_min
            rect_nms["x2"] = x_max
            rect_nms["y1"] = y_min
            rect_nms["y2"] = y_max
            nms_rectangles.append(rect_nms)

            rectangles = [r for r in rectangles if r not in similiar]

    # svakom od pravugaonika doda sliku kako mogao SVM da klasifikuje slovo regiona
    for nms_rectangle in nms_rectangles:
        y_min = nms_rectangle['y1']
        y_max = nms_rectangle['y2']
        x_min = nms_rectangle['x1']
        x_max = nms_rectangle['x2']
        nms_rectangle["image"] = class_image[y_min: y_max + 1, x_min: x_max + 1]

    return nms_rectangles


def extract_rows_OCR(class_image):
    rows = []
    distances = []

    # @TODO: ovde biras jednu od metoda kojom izdavajamo regione oko slova
    # mozes dodavati nove metode ili menjati postojece...

    # regions_array = make_OCR_rectangles_old_roi(class_image)
    regions_array = make_OCR_rectangles(class_image)

    # sortiras ih po donjem desnom y-u da bi mogao da izdvojis redove
    regions_array = sorted(regions_array, key=lambda x: x["y2"])

    j = 0
    i = 0
    while i < len(regions_array):
        row = []
        # koristi da se odredi sa K-means sta je razlika izmedju slova, a sta izmedju reci u jednom redu
        row_distances = []
        first_row_rect = regions_array[i]
        y_row = first_row_rect["y2"]

        for character in regions_array[i:]:
            character_bottom_y = character["y2"]
            j += 1
            if abs(character_bottom_y - y_row) < 30:
                row.append(character)
            else:
                j -= 1
                #  od tog pocinje novi red
                break

        # izdvojen je ceo jedan red
        row = sorted(row, key=lambda x: x["x1"])
        if row[len(row)-1]["x1"] > row[len(row)-2]["x1"]*1.7:
            row.pop(len(row)-1)

        # da preskocis sve konture koje su u tom redu da njih ne gleda petlja
        i = j

        x1 = min([r["x1"] for r in row])
        y1 = min([r["y1"] for r in row])
        x2 = max([r["x2"] for r in row])
        y2 = max([r["y2"] for r in row])
        new_row = dict()
        new_row['x1'] = x1
        new_row['x2'] = x2
        new_row['y1'] = y1
        new_row['y2'] = y2
        new_row["image"] = class_image[y1: y2 + 1, x1: x2 + 1]
        # izracunaj distance u svakom row izmedju kontura.
        # row_distances = calculate_row_distances(row)
        # distances += row_distances
        rows.append(new_row)

    # distances = np.asarray(distances).reshape(-1, 1)
    # k_means = KMeans(n_clusters=2, random_state=0).fit(distances)
    return rows, regions_array


def process_row_OCR(row, svm_char):
    words = []
    word = ""
    # distances = calculate_row_distances(row)
    # ovo je da posle poslednje konture u redu nista ne znaci
    # koje je rastojanje a da ne moram da petljam sa indeksima
    # distances.append(-100)
    # distance_types = kmeans.predict(np.asarray(distances).reshape(-1, 1))

    # for region_indx, row_region in enumerate(row):
        # img_words_gs = image_gray(row_region[0])
        # img_words_bin = image_bin(img_words_gs)
        # img_words_bin = invert(img_words_bin)
        # image = resize_region_OCR(make_RBG_from_binary_image(img_words_bin))

        # row_region["image"] = resize_region_OCR(row_region["image"])
        # char_string = predict_char(row_region["image"], svm_char)
        # plt.imshow(row_region["image"])
    return read_char(row["image"])

        # plt.imshow(row_region["image"])
        # plt.show()
        # print("prepoznao: " ,char_string[0] )

        # dist_type = distance_types[region_indx]
        # da li ce rastojanje izmedju reci uvek da bude 1
        # a rastojanje unutar reci da bude 0 u kmeans..
        # if dist_type == 1:
        #     words.append(word)
        #     word = ""

    # if word not in words:
    # words.append(word)

    # return words


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


def is_inside(bb1, bb2):
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # if bb1['x1'] - bb2['x1'] and bb1['y1'] <= bb2['y1'] and bb1['x2'] >= bb2['x2'] and bb1['y2'] >= bb2['y2']:
    #     return True
    # elif bb2['x1'] <= bb1['x1'] and bb2['y1'] <= bb1['y1'] and bb2['x2'] >= bb1['x2'] and bb2['y2'] >= bb1['y2']:
    #     return True

    if bb2_area <= bb1_area and (intersection_area / bb2_area) > 0.9:
        return True
    elif bb1_area <= bb2_area and (intersection_area / bb1_area) > 0.9:
        return True
    else:
        return False


def perform_class_OCR(OCR_region, index):
    class_image = cv2.resize(OCR_region[0], (300, 300), interpolation=cv2.INTER_AREA)
    # class_image = cv2.bilateralFilter(class_image, 10, 60, 60)
    # char_svm = load_svm_char_bin()
    char_svm = load_svm_char()

    rows, all_regions = extract_rows_OCR(class_image)
    text_array = []
    for indx, row in enumerate(rows):
        row_words = read_char(row["image"])
        text_array.append(row_words)
        # print("slika: ", index, " index reda: ", indx, " procitao: ", row_words)
    print("****")
    for reg in all_regions:
        (x1, y1, x2, y2) = (reg["x1"], reg["y1"], reg["x2"], reg["y2"])
        cv2.rectangle(class_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.imshow(class_image)
    plt.show()

    generated_class = Class(text_array, OCR_region)
    return generated_class
