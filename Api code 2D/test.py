from faceAntiSpoof2D import faceAntiSpoof2D
import cv2

_faceAntiSpoof2D = faceAntiSpoof2D(modelFile = 'model/best')

test_img = cv2.imread('test_img.jpg')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

print(_faceAntiSpoof2D.detect(test_img))
