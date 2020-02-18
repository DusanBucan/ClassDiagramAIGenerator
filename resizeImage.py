import cv2
import numpy as np

def image_bin_f(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def apativeTreshHold(greyImage):
    return cv2.adaptiveThreshold(greyImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 25)


def resizeImage(img, newSize):
    greyImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image_bin = apativeTreshHold(greyImage)

    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = None
    optimal_area = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 900000:
            if optimal_area is None or optimal_area < area:
                optimal_area = area
                main_contour = contour
#             kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
#             oznaciti region pravougaonikom na originalnoj slici sa rectangle funkcijom
    x, y, w, h = cv2.boundingRect(main_contour) # koordinate i velicina granicnog pravougaonika
    region = img[y:y+h+1, x:x+w+1]
    a = cv2.resize(region, (newSize[1], newSize[0]), interpolation=cv2.INTER_NEAREST)
    return a.astype(np.float32)
