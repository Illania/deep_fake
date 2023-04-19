from insightface_func.face_detect_crop_multi import Face_detect_crop


class FaceDetectAntelopeModel(Face_detect_crop):

    def __init__(self, threshold, size):
        super().__init__(name='antelope', root='./insightface_func/models')
        super().prepare(ctx_id=0, det_thresh=threshold, det_size=(size, size))
