import cv2
import pytesseract
import argparse
from util_func import find_date_fields

"""
Find date fields using OpenCV2 find counters and OCR only
"""

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(
    gray, 110, 255, cv2.THRESH_BINARY_INV)

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

im2 = image.copy()

text_list = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    rect = cv2.rectangle(im2, (x, y), (x + y, y + h), (0, 255, 0), 2)
    cropped = im2[y:y + h, x:x + w]

    text = pytesseract.image_to_string(cropped)

    if not text.isspace():
        text_list.append(text)

print(find_date_fields(text_list))
