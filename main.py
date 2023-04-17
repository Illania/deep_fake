import os
from FaceDetectAntelopeModel import FaceDetectAntelopeModel
from SwapManager import SwapManager


if __name__ == "__main__":
    os.chdir("SimSwap")
    app = FaceDetectAntelopeModel(0.6, 640)
    swapManager = SwapManager(app, '../demo/input.mp4',
                                   '../demo/output/output.mp4',
                                   '../demo/multispecific')
    swapManager.swap()
