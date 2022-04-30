import cv2
import streamlit as st
import tempfile

st.title("Face-Anti spoofing - DEMO")

# set Webcam
camera = cv2.VideoCapture(0)

# Set Video
video_upload = st.file_uploader("Choose a file")
tfile = tempfile.NamedTemporaryFile(delete=False) 

if video_upload is not None:
    tfile.write(video_upload.read())

video = cv2.VideoCapture(tfile.name)

col1, col2= st.columns([1, 4])


with col1:
    st.header('Mode')
    genre = st.radio( '',('Video', 'Webcam'))

with col2:
    FRAME_WINDOW = st.image([])

while genre == "Webcam":
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)


while genre == "Video" and video.isOpened():
    _, frame = video.read()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    
col1, col2, col3 = st.columns(3)
col2.metric("2D", "Live", "-spoof")
col3.metric("3D", "Live", "+live")