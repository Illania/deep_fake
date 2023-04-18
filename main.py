import os
from FaceDetectAntelopeModel import FaceDetectAntelopeModel
from SwapManager import SwapManager

if __name__ == "__main__":
    os.chdir("SimSwap")

    app = FaceDetectAntelopeModel(0.6, 640)
    swapManager = SwapManager(app)

    swapManager.swap_single('../demo/single/input.mp4',
                            '../demo/single/results/result.mp4',
                            '../demo/single/dst.jpeg')

    swapManager.swap_multi('../demo/multi/input.mp4',
                           '../demo/multi/results/output.mp4',
                           '../demo/multi/multispecific')
