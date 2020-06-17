import os
import cv2

from PIL import Image
import pytesseract
import matplotlib.pyplot as plt


def read_char(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(image, None, fx=0.5, fy=0.5)
    # 3. Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check to see if we should apply thresholding to preprocess the
    # image
    # gray = cv2.threshold(gray, 0, 255,
    #                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # make a check to see if median blurring should be done to remove
    # noise
    # gray = cv2.medianBlur(gray, 3)
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    # plt.imshow(gray, 'gray')
    # plt.show()
    # gray = cv2.medianBlur(gray, 3)
    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    # filename = "{}.png".format(os.getpid())
    # cv2.imwrite(filename, gray)

    # ne znam da li ovo treba na linuxu
    # ako treba namesti putanju, ako ne zakomentarisi
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # 7 posmatra sliku kao red reci  nije bas toliko dobro
    # 8 posmatra sliku kao jednu rec
    # 9 posmatra sliku kao jednu rec u krug tako nesto
    # 8,9 daju iste rezultate, za sad najbolje, pokusala sam da menjam sliku, smanjujem, treshold, blur
    # nije bas sjajno, ovako je za sad najbolje
    config = "--psm 9"
    text = pytesseract.image_to_string(gray, config=config)
    # os.remove(filename)
    # print(text)
    # show the output images
    # cv2.waitKey(0)
    # class_array.append(c)
    return text
