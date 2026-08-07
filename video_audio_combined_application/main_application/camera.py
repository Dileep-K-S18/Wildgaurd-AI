import cv2
import torch
import pathlib
from gtts import gTTS
from pydub import AudioSegment
import pygame
import os
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class VideoCamera:
    def __init__(self):
        self.model = torch.hub.load('.', 'custom', path='last_new.pt', source='local')
        self.model.conf = 0.10
        self.model.iou = 0.25
        self.cap = None

    def start(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

    def stop(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        results = self.model(frame)
        boxes = results.pandas().xyxy[0]

        for i in range(len(boxes)):
            xmin, ymin = int(boxes.iloc[i]['xmin']), int(boxes.iloc[i]['ymin'])
            xmax, ymax = int(boxes.iloc[i]['xmax']), int(boxes.iloc[i]['ymax'])
            label = str(boxes.iloc[i]['name'])
            conf = float(boxes.iloc[i]['confidence'])

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


            text = str(label)
            language = 'en'


            # Create gTTS object
            tts = gTTS(text=text, lang=language, slow=False)

            # Save the converted audio in a file
            tts.save("output.mp3")
            print("Audio file saved successfully.")
        
            pygame.mixer.init()

            # Load the audio file
            pygame.mixer.music.load("output.mp3")

            # Play the audio file
            pygame.mixer.music.play()

            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)  # Adjust the tick value as needed

            # Close the mixer
            pygame.mixer.quit()

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
