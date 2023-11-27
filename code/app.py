import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #
    st.title("Human Emotion Detection System")
    activiteis = ["Home", "Webcam Face Detection"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """    
            """)
    if choice == "Home":

        
        html_temp_home1 = """<div style="background-color:#3399ff;padding:10px">
                                            <h2 style="color:white;text-align:center;">
                                            Real time face emotion recognization.</h2>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                Welcome to our cutting-edge platform for real-time face emotion recognition. Our technology harnesses the power of artificial intelligence to analyze and understand human emotions as they happen, providing a fascinating glimpse into the world of human expression. With our user-friendly interface, you can experience the magic of instantaneous emotion detection, from joy to sadness, surprise to anger, and everything in between. 
                Whether you're a researcher, developer, or simply curious about the possibilities of emotion recognition, our platform offers a unique window into the captivating realm of human emotions in real time.
                

                 """)
        
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)

    # elif choice == "About":
    #     st.subheader("About this app")
    #     html_temp_about1= """<div style="background-color:#3399ff;padding:10px">
    #                                 <h4 style="color:white;text-align:center;">
    #                                 Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
    #                                 </div>
    #                                 </br>"""
    #     st.markdown(html_temp_about1, unsafe_allow_html=True)

    #     html_temp4 = """
    #                          		<div style="background-color:#98AFC7;padding:10px">
    #                          		<h4 style="color:white;text-align:center;">This Application is developed by Abhinav Nagar using Streamlit Framework, Opencv, Tensorflow and Keras library  </h4>
    #                          		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
    #                          		</div>
    #                          		<br></br>
    #                          		<br></br>"""

    #     st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()
