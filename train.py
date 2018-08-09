import cv2
import numpy as np
from PIL import Image
import os

# Source for code used in this project: https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if os.path.exists('./model.yml'):
    input(
        "The model file already exists. Press enter if you would like to overwrite the current model and proceed."
    )


def prepare_data(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    samples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return samples, ids


faces, ids = prepare_data(path)
recognizer.train(faces, np.array(ids))

recognizer.write('model.yml')

print("\n{0} faces trained. Exiting Program".format(len(np.unique(ids))))
