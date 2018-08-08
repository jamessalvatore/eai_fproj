import cv2
import numpy as np
import os

from picamera.array import PiRGBArray
from picamera import PiCamera

import time
from util import *


def main():
    approved = {}
    unknowns = []

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    #iniciate id counter
    id = 0
    # names related to ids: example ==> Marcelo: id=1,  etc
    # names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']
    names = get_contacts()
    # Initialize and start realtime video capture

    cam = PiCamera()
    cam.resolution = (640, 480)
    cam.framerate = 32
    rawCapture = PiRGBArray(cam, size=(640, 480))

    time.sleep(.1)

    names = get_contacts()
    if names == {}:
        print('No contact names found in your contacts file. Exiting...')
        return

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    user_times = {}

    #while True:
    for frame in cam.capture_continuous(
            rawCapture, format="bgr", use_video_port=True):
        #ret, img = cam.read()
        # img = cv2.flip(img, -1)  # Flip vertically
        img = frame.array
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            # confidence here is the distance to the closest item (calculated using chi-square distance) in the dataset that matches
            # this id. 0 is a perfect match.
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))

                image_file = 'captured.png'
                cv2.imwrite(image_file, img)

                if id in approved:
                    if (time.time() - approved[id]) > 10:
                        send_email(image_file, "approved")
                    else:
                        # Do we want to send an email for this?
                        approved[id] = time.time()
                else:
                    approved[id] = time.time()
                    send_email(image_file, "approved")
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

                image_file = 'captured.png'
                cv2.imwrite(image_file, img)

                if len(unknowns) > 99:
                    unknowns.remove(unknowns[0])
                unknowns.append(time.time())
                if unknowns[len(unknowns) - 1] > 10:
                    send_email(image_file, "unkown")

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255),
                        2)
            # dont show the confidence if it's unknown (not relevant)
            # cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1,
            #             (255, 255, 0), 1)

        #cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

        rawCapture.truncate(0)
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
