# Importing kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Importing other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Building app and layout
class CamApp(App):
    def build(self):
        # Main Layout component
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify",on_press= self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated",size_hint=(1,.1))

        # Adding items to Layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam) 
        layout.add_widget(self.verification_label)
        layout.add_widget(self.button)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('siamesemodel2.h5',custom_objects={'L1Dist':L1Dist})

        # Setting up video capturing device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update,1.0/33.0)


        return layout
    
    # will run continuously to get web cam feed
    # running 1 time in every 33 sec to update our webcam
    def update(self, *args):

        # reading frame from opencv
        ret, frame = self.capture.read()
        # Optional: Flip horizontally like a mirror
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape
        box_size = 300
        x_center = width // 2
        y_center = height // 2

        # Shift ROI slightly to the left (-60)
        x1 = x_center - box_size // 2 - 60
        y1 = y_center - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        roi = frame[y1:y2, x1:x2]
        


        # converting img to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Preprocessing image
    # preproccessing - Scale and Resize
    def preprocess(self, file_path):

        # read image from file path
        byte_img = tf.io.read_file(file_path)

        # loading image
        img = tf.io.decode_jpeg(byte_img)

        # preprocessing(resizing image to 100x100x3
        img = tf.image.resize(img,(100,100))
        img = img / 255.0
        return img
    
    # verification function
    def verify(self, *args):
        # specifying threshold
        detection_threshold = 0.5 # limit before our prediction is considered positive
        verification_threshold = 0.5 # what proposition of predictions need to be positive for a match

        # Capturing input image from webcam
        SAVE_PATH = os.path.join('application_data','input_image','input_image.jpg')
        # reading frame from opencv
        ret, frame = self.capture.read()
        # Optional: Flip horizontally like a mirror
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape
        box_size = 300
        x_center = width // 2
        y_center = height // 2

        # Shift ROI slightly to the left (-60)
        x1 = x_center - box_size // 2 - 60
        y1 = y_center - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        roi = frame[y1:y2, x1:x2]
        cv2.imwrite(SAVE_PATH,frame)
        

        # Building results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_image')):
            if image.endswith('.jpg') or image.endswith('.jpeg') or image.endswith('.png'):
                input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
                validation_img = self.preprocess(os.path.join('application_data', 'verification_image', image))

                # Make Predictions 
                result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
                results.append(result)

        # Detection Threshold: Metric above which a prediction is considered positive
        detection = np.sum(np.squeeze(results) > detection_threshold)

        # Verification Threshold: Proportion of positive predictions / total
        verification = detection / len(results)
        verified = verification > verification_threshold
    

        # setting verification text
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'
    
            # log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
            
        return results, verified

if __name__ == '__main__':
    CamApp().run()


