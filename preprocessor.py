import cv2
import numpy as np
from mtcnn import MTCNN
from numpy import expand_dims
import pandas as pd
import os
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from FaceDetectors.face_detector import FaceDetector
from FaceDetectors.mtcnn_face_detector import MtcnnFaceDetector


class Preprocessor:
    """ Performs preprocessing operations on videos. """

    __detector = MtcnnFaceDetector()

    def __init__(self, detector):
        self.detector = detector

    def extract_faces(self, source, destination):
        """ Extract faces from all images in the source folder and save face images to the destination folder. """
        counter = 0
        for directory, _, filenames in os.walk(source):
            for filename in filenames:
                try:
                    image = cv2.imread(os.path.join(directory, filename))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face = self.__detector.detect_face(image)
                    face = cv2.resize(face, (120, 120), interpolation=cv2.INTER_LINEAR)
                    plt.imsave(os.path.join(destination, str(counter) + '.jpg'), face)
                    print('Saved: ', os.path.join(destination, str(counter) + '.jpg'))
                except:
                    pass
                counter += 1
