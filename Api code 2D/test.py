from faceAntiSpoof2D import faceAntiSpoof2D
import cv2

_faceAntiSpoof2D = faceAntiSpoof2D()

test_img = cv2.imread('test_img.jpg')

print(_faceAntiSpoof2D.detect(test_img))
