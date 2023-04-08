import dlib
import cv2
from FaceDetectors.face_detector import FaceDetector


class DlibFaceDetector(FaceDetector):

    def detect_face(self, image_path):
        detector = dlib.get_frontal_face_detector()
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = detector(gray, 1)[0]
        x = face.left()
        y = face.top()
        x_ = face.right()
        y_ = face.bottom()
        x1, y1, x2, y2 = x - 10, y + 10, x_ - 10 + 20, y_ + 10
        return image[y1:y2, x1:x2]
