import cv2
import numpy as np
import onnxruntime as rt

class faceLandmark:
    def __init__(self, modelFile):
        sessOptions = rt.SessionOptions()
        sessOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL 
        self.landmarkDetector = rt.InferenceSession(modelFile, sessOptions)

    def preprocessInput(self, inputImage, faceBoxes):

        inputImageRGB = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)

        height,width,_ = inputImageRGB.shape

        preprocessFaces = []
        scaleFaceBoxes = []

        for faceBox in faceBoxes:
            x1, y1, x2, y2, prob = faceBox.astype(np.int32)

            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(max([w, h])*1.1)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = x1 + size
            y1 = cy - size//2
            y2 = y1 + size
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)  

            croppedFace = inputImageRGB[y1:y2,x1:x2,:]
            scaleFaceBoxes.append([x1, y1, x2, y2])

            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):   
                croppedFace = cv2.copyMakeBorder(croppedFace, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)  
            croppedFace = cv2.resize(croppedFace, (112, 112))

            preprocessFaces.append(croppedFace)
        
        preprocessFaces = np.array(preprocessFaces).astype(np.float32)
        preprocessFaces = preprocessFaces / 255.0
        preprocessFaces = preprocessFaces.transpose(0,3,1,2)
        scaleFaceBoxes = np.array(scaleFaceBoxes)

        return preprocessFaces, scaleFaceBoxes

    def remapLandmark(self, netOutputs, faceBoxes):
        remapLandmarks = []
        for faceIdx, netOutput in enumerate(netOutputs):
            faceX1, faceY1, faceX2, faceY2 = faceBoxes[faceIdx].astype(np.int32)
            faceW, faceH = faceX2 - faceX1, faceY2 - faceY1
            landmark = netOutput.reshape(-1, 2)
            landmark[:, 0] = landmark[:, 0] * faceW + faceX1
            landmark[:, 1] = landmark[:, 1] * faceH + faceY1
            remapLandmarks.append(landmark)

        remapLandmarks = np.array(remapLandmarks)
        return remapLandmarks
    def extractLandmark(self, inputImage, faceBoxes):

        preprocessFaces, scaleFaceBoxes = self.preprocessInput(inputImage, faceBoxes)

        netOutputs = self.landmarkDetector.run([], {'input': preprocessFaces.copy()})[0]
        remapLandmarks = self.remapLandmark(netOutputs, scaleFaceBoxes)
        return remapLandmarks