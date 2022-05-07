import cv2
import numpy as np
import sys, os
from depthEstimationModel_2d import DepthEstimationModel

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/faceDetetorAndAlignment")
from faceDetectorAndAlignment import faceDetectorAndAlignment

class faceAntiSpoof2D:
    def __init__(self, modelFile = 'model/best'):
        self.modelFile = modelFile
        self._faceDetectorAndAlignment = faceDetectorAndAlignment(os.path.dirname(os.path.realpath(__file__))
                                                           + '/faceDetetorAndAlignment/models/faceDetectorV2.onnx', processScale=0.20)

        self.model = DepthEstimationModel()
        self.model.compile()
        self.model.load_weights(self.modelFile).expect_partial()

        self.threshold = 0.8

    def detect(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        croppedImage, faceBoxes = self.cropImage(image)
        
        if len(faceBoxes) == 0:
            return False, 0

        predict = self.estimate(croppedImage)

        score = predict[0][0][0]
        #print(score)
        return score >= self.threshold, score

    def detectAfterPreprocess(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predict = self.estimate(image)

        score = predict[0][0][0]
        #print(score)
        return score >= self.threshold, score

    def cropImage(self, image):
        while image.shape[0] < 500 or image.shape[1] < 500:
            image = cv2.resize(image, (0,0), fx = 2, fy = 2)

        faceBoxes, _, _ = self._faceDetectorAndAlignment.detect(image)
        crop_Image = image.copy()
        
        

        for faceBox in faceBoxes:
            x1,y1,x2,y2,_ = faceBox.astype(np.int32)
            crop_Image = crop_Image[y1:y2, x1:x2]

        return crop_Image, faceBoxes

    def estimate(self, croppedImage):
        croppedImage = croppedImage/255
        finalImage = croppedImage.reshape(1, croppedImage.shape[0], croppedImage.shape[1], 3)

        predict = self.model.predict(finalImage)
        return predict
