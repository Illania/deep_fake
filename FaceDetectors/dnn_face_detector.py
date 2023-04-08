from FaceDetectors.face_detector import FaceDetector
import cv2
import numpy as np


class DnnFaceDetector(FaceDetector):
    def detect_face(self, image_path):
        model_file = "dnn_models/res10_300x300_ssd_iter_140000.caffemodel"
        config_file = "dnn_models/deploy.prototxt.txt"
        net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces = net.forward()
        box = faces[0, 0, 0, 3:7] * np.array([w, h, w, h])
        (x, y, x_, y_) = box.astype("int")
        x1, y1, x2, y2 = x - 10, y + 10, x_ - 10 + 20, y_ + 10
        return image[y1:y2, x1:x2]
