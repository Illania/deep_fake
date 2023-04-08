import cv2
from FaceDetectors.face_detector import FaceDetector


class HaarCascadeFaceDetector(FaceDetector):

    def detect_face(self, image_path):
        classifier = cv2.CascadeClassifier('haar_models/haarcascade_frontalface2.xml')
        image = cv2.imread(image_path)
        face = classifier.detectMultiScale(image)[0]
        x, y, w, h = face
        x_, y_ = x + w, y + h
        x1, y1, x2, y2 = x - 10, y + 10, x_ - 10 + 20, y_ + 10
        return image[y1:y2, x1:x2]
