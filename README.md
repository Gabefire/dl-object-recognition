# Driver License Models

Project to practice OCR and OpenCV. Used two machine learning models to gather different information from driver's license test image

## EAST Text Detection

Ran driver's license through EAST Text Detection model to get region's of interest for all the text fields then ran each ROI through pytesseract(OCR) to turn text to string. Ran text strings through a regular expression to grab strings that contain dates.

```
git clone https://github.com/Gabefire/dl-object-recognition.git
cd dl-object-recognition
python3 text_detection.py --image test-image.jpeg
```

## KNN classifier

Trained model on k nearest neighbors data located in the train folder. Evaluated accuracy with images being converted to raw pixels and images being converted to a histogram Once model was trained compared it to test-image.jpeg to figure out the most likely state.

```
git clone https://github.com/Gabefire/dl-object-recognition.git
cd dl-object-recognition
python3 knn_classifier.py -d ./train/ -k 3
```
