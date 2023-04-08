from FaceDetectors.face_detector import FaceDetector
import cv2
from mtcnn.mtcnn import MTCNN


class MtcnnFaceDetector(FaceDetector):
    def detect_face(self, image_path):
        detector = MTCNN()
        image = cv2.imread(image_path)
        faces = detector.detect_faces(image)
        x, y, w, h = faces[0]['box']
        x1, y1, x2, y2 = x - 10, y + 10, x - 10 + w + 20, y + 10 + h
        return image[y1:y2, x1:x2]
