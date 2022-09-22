import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
from keras.models import load_model
import webbrowser
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('b.jpg')  
model=load_model('model_file_30epochs.h5')
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

try:
    emotion=np.load("emotion.npy")
except:
    emotion=""


class EmotionProcessor:
    def recv(self,frm):
        frame=frm.to_ndarray(format="bgr24")
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces= faceDetect.detectMultiScale(gray, 1.3, 3)
        for x,y,w,h in faces:
            sub_face_img=gray[y:y+h, x:x+w]
            resized=cv2.resize(sub_face_img,(48,48))
            normalize=resized/255.0
            reshaped=np.reshape(normalize, (1, 48, 48, 1))
            result=model.predict(reshaped)
            label=np.argmax(result, axis=1)[0]
            print(label)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            np.save("emotion.npy",np.array([labels_dict[label]]))

        return av.VideoFrame.from_ndarray(frame,format="bgr24")

st.title("Emotion Based Music Player")
nav=st.radio("Mode",["Enter Mood Manually","Capture Video"])
if nav=="Capture Video":
    webrtc_streamer(key="key",desired_playing_state=True,video_processor_factory=EmotionProcessor)
    btn=st.button("Recommend me songs")

    if btn:
        if not(emotion):
            st.warning("Please let me capture your emotion first")
        else:
            webbrowser.open(f"https://open.spotify.com/search/{emotion[0]}%20songs")
    
else:
    Emotion = st.text_input("Emotion")
    if Emotion not in ['','Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise','angry','disgust','fear','happy','neutral','sad','surprise','ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']:
        st.warning("Please enter the correct Emotion")
    else:
        emotion[0]=Emotion
        btn=st.button("Recommend me songs")

        if btn:
           webbrowser.open(f"https://open.spotify.com/search/{emotion[0]}%20songs")

#streamlit run music.py
