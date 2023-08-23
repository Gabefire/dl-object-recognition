
import re
import numpy as np
import cv2
import time
import argparse
from imutils.object_detection import non_max_suppression
import pytesseract
from operator import itemgetter

"""
Run the EAST deep learning model and then run ID through tesseract to pull dates
"""


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())


def east_detect(image):
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    orig = image.copy()

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height: Should be multiple of 32
    (newW, newH) = (320, 320)

    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))

    (H, W) = image.shape[:2]

    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)

    start = time.time()

    net.setInput(blob)

    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            # Set minimum confidence as required
            if scoresData[x] < 0.5:
                continue
                # compute the offset factor as our resulting feature maps will
            #  x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)
    boxes = sorted(boxes, key=itemgetter(1, 0))
    image_rois = []
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)-10
        startY = int(startY * rH)-10
        endX = int(endX * rW)+10
        endY = int(endY * rH)+10
        image_rois.append(orig[startY:endY, startX:endX])

    print(time.time() - start)
    return orig, image_rois


image = cv2.imread(args["image"])
(out_image, image_rois) = east_detect(image)

text_list = []
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 20, 225,
                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


for img in image_rois:
    text_list.append(pytesseract.image_to_string(img))


re_list = []
for text in text_list:
    if re.search("([0-9]){1,2}\/([0-9]){1,2}\/([0-9]){2,4}", text):
        text.strip()
        text = re.search(r"([0-9]){1,2}\/([0-9]){1,2}\/([0-9]){2,4}", text)
        re_list.append(text.group())

print(re_list)
