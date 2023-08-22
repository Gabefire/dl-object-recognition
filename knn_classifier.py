import pickle
import cv2
import os
import argparse
import imutils
from imutils import contours
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import pytesseract


def image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    else:
        cv2.normalize(hist, hist)

    return hist.flatten()


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs for k_nn distance (-1 uses all available cores)")
args = vars(ap.parse_args())

print("describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

rawImages = []
features = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)

    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)

    print("processed {}/{}".format(i, len(imagePaths)))

rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("pixels matrix: {:.2f}MB".format(rawImages.nbytes/(1024*1000.0)))
print("features matrix: {:.2f}MB".format(features.nbytes/(1024*1000.0)))

(trainRI, testRI, trainRL, testRL) = train_test_split(
    rawImages, labels, test_size=0.25, random_state=42
)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
    features, labels, test_size=0.25, random_state=42
)


print("evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
                             n_jobs=args["jobs"])
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("histogram accuracy: {:.2f}%".format(acc*100))

print("evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
                             n_jobs=args["jobs"])
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("raw pixel accuracy: {:.2f}%".format(acc * 100))

# test image
test_image = cv2.imread("test-image.jpeg")

test_vector = image_to_feature_vector(test_image)
test_vector = np.array(test_vector)
print(model.predict(test_vector.reshape(1, -1)))
