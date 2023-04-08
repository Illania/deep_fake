from abc import ABCMeta, abstractmethod


class FaceDetector(metaclass=ABCMeta):
    """ Base face detector class. """

    @abstractmethod
    def detect_face(self, image_path):
        pass
