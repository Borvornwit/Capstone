import cv2
import streamlit as st
import tempfile
import sys, os
import threading
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/Api code 2D')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/Api code 3D')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/faceDetetorAndAlignment')
from faceAntiSpoof2D import faceAntiSpoof2D
#from faceAntiSpoof2D_old import faceAntiSpoof2D_old
from m3Dapi import FAS3D
from faceDetectorAndAlignment import faceDetectorAndAlignment
from faceLandmark import faceLandmark

_faceAntiSpoof2D = faceAntiSpoof2D(modelFile = 'Api code 2D/model/best')
#_faceAntiSpoof2D = faceAntiSpoof2D_old(modelFile = 'Api code 2D/model/old')
_faceAntiSpoof3D = FAS3D(modelFile='Api code 3D/Tranrppg10sec new model_drop_5.pt')
_faceLandmarks = faceLandmark('faceDetetorAndAlignment/models/faceLandmark64Light.onnx')

st.title("Face-Anti spoofing - DEMO")

# set Webcam
camera = cv2.VideoCapture(0)

# Set Video
video_upload = st.file_uploader("Choose a file")
tfile = tempfile.NamedTemporaryFile(delete=False)

if video_upload is not None:
    tfile.write(video_upload.read())

video = cv2.VideoCapture(tfile.name, cv2.CAP_FFMPEG)

col1, col2= st.columns([1, 4])

with col1:
    st.header('Mode')
    genre = st.radio( '',('Video', 'Webcam'))

with col2:
    FRAME_WINDOW = st.image([])

def strResult(result):
    if result is None:
        result_str = 'Waiting...'
    elif result:
        result_str = 'Live'
    else:
        result_str = 'Spoof'
    return result_str

def updateResult():
    # result_2d_str = "Live" if result_2d else "Spoof"
    # result_3d_str = "Live" if result_3d else "Spoof"
    result_2d_str = strResult(result_2d)
    result_3d_str = strResult(result_3d)

    with placeholder.container():
        col1, col2, col3 = st.columns(3)
        col2.metric("2D", result_2d_str, str(score_2d))
        col3.metric("3D", result_3d_str)

def preprocess(frame):
    global alignedImage, faceBoxes, faceLandmarks
    alignedImage, faceBoxes = _faceAntiSpoof2D.cropImage(frame)
    faceLandmarks = _faceLandmarks.extractLandmark(frame, faceBoxes)
    # print('preprocess',faceLandmarks)

def predict2d(frame):
    global result_2d, score_2d
    result_2d, score_2d = _faceAntiSpoof2D.detectAfterPreprocess(frame)

def predict3d(final_mstmap_face,final_mstmap_bg):
    global result_3d, frame_count
    print(frame_count,result_3d)
    if final_mstmap_face is None: 
        result_3d = None
        return
    print('final_mstmap_face',final_mstmap_face.shape)
    result_3d = _faceAntiSpoof3D.predict(final_mstmap_face,final_mstmap_bg)

def faceAntiSpoof(frame):
    global thread_preprocess, thread_2d, thread_3d
    global frame_count, faceLandmarks

    # resize frame
    frame = cv2.resize(frame, (0,0), fx = 2, fy = 2)

    # preprocess
    if not thread_preprocess.is_alive():
        thread_preprocess = threading.Thread(target=preprocess, args=(frame,))
        thread_preprocess.start()

    # if face not detected
    if len(faceBoxes) == 0:
        frame_count = 0
        return

    # 2DFAS
    if not thread_2d.is_alive():
        thread_2d = threading.Thread(target=predict2d, args=(alignedImage,))
        thread_2d.start()

    # 3DFAS
    if faceLandmarks is None:
        return
    # print('FAS',faceLandmarks)
    h , w,_ = frame.shape
    x1,y1,x2,y2,_ = faceBoxes[0].astype(np.int32)
    bgLandmarks = [x1,y1,x2,y2,w,h]
    final_mstmap_face,final_mstmap_bg = _faceAntiSpoof3D.genFromSeq(frame,faceLandmarks,bgLandmarks,frame_count,cam_id=0) 
    frame_count += 1

    if not thread_3d.is_alive():
        thread_3d = threading.Thread(target=predict3d, args=(final_mstmap_face,final_mstmap_bg,))
        thread_3d.start()

thread_preprocess = threading.Thread(target=preprocess)
thread_2d = threading.Thread(target=predict2d)
thread_3d = threading.Thread(target=predict3d)

alignedImage = None
faceBoxes = []
faceLandmarks = None

result_2d = None
score_2d = 0.0

result_3d = None
frame_count = 0

placeholder = st.empty()

while genre == "Webcam" and camera.isOpened():
    _, frame = camera.read()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        faceAntiSpoof(frame)
        updateResult()
    else:
        break

camera.release()


while genre == "Video" and video.isOpened():
    _, frame = video.read()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        faceAntiSpoof(frame)
        updateResult()
    else:
        break

video.release()
