import cv2
import streamlit as st
import tempfile
import sys, os
import threading

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/Api code 2D')
from faceAntiSpoof2D import faceAntiSpoof2D
#from faceAntiSpoof2D_old import faceAntiSpoof2D_old

_faceAntiSpoof2D = faceAntiSpoof2D(modelFile = 'Api code 2D/model/best')
#_faceAntiSpoof2D = faceAntiSpoof2D_old(modelFile = 'Api code 2D/model/old')

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

def updateResult():
    result_2d_str = "Live" if result_2d else "Spoof"
    result_3d_str = "Live" if result_3d else "Spoof"

    with placeholder.container():
        col1, col2, col3 = st.columns(3)
        col2.metric("2D", result_2d_str, str(score_2d))
        col3.metric("3D", result_3d_str)

def predict2d(frame):
    global result_2d, score_2d
    result_2d, score_2d = _faceAntiSpoof2D.detectAfterPreprocess(frame)

result_2d = False
score_2d = 0.0
result_3d = False
thread_2d = threading.Thread(target=predict2d)

placeholder = st.empty()

while genre == "Webcam" and camera.isOpened():
    _, frame = camera.read()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        updateResult()

        alignedImage, faceBox = _faceAntiSpoof2D.cropImage(frame)

        if not thread_2d.is_alive():
            thread_2d = threading.Thread(target=predict2d, args=(alignedImage,))
            thread_2d.start()
    else:
        break

camera.release()


while genre == "Video" and video.isOpened():
    _, frame = video.read()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        updateResult()

        alignedImage, faceBox = _faceAntiSpoof2D.cropImage(frame)

        if not thread_2d.is_alive():
            thread_2d = threading.Thread(target=predict2d, args=(alignedImage,))
            thread_2d.start()
    else:
        break

video.release()
