import cv2
import numpy as np
from faceDetectorAndAlignment import faceDetectorAndAlignment
from faceLandmark import faceLandmark
_faceDetectorAndAlignment = faceDetectorAndAlignment('models/faceDetectorV2.onnx', processScale=0.20) # increase processScale will increase detection accuracy but reduce fps
_faceLandmarks = faceLandmark('models/faceLandmark64Light.onnx')

videoStream = cv2.VideoCapture(0)
while True:
    isValidFrame, inputFrame = videoStream.read()
    
    if isValidFrame:
        faceBoxes, faceLandmarksFivePts, alignedFaces = _faceDetectorAndAlignment.detect(inputFrame)
        if len(faceBoxes) > 0:
            faceLandmarks = _faceLandmarks.extractLandmark(inputFrame, faceBoxes)


            for faceBox in faceBoxes:
                x1,y1,x2,y2,_ = faceBox.astype(np.int32)
                cv2.rectangle(inputFrame, (x1,y1), (x2,y2), (0,255,0),3)

            for currentFaceLandmark in faceLandmarks:
                for pts in currentFaceLandmark:
                    x, y = pts.astype(np.int32)
                    cv2.circle(inputFrame, (x, y), 2, (0,255,0), -1)
        

        cv2.imshow("Output", inputFrame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

videoStream.release()