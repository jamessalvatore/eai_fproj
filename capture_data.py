import cv2
import json
import os
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

from util import get_contacts

# Source for code used in this project: https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826


def main():

    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))

    time.sleep(.1)

    face_detector = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')

    u_id = input('Enter the id for this user: ')

    contacts = get_contacts()

    if (u_id in contacts):
        print('Updating data for user: [' + u_id + '] ' + contacts[(u_id)])
    else:
        name = input('Enter a name for this user: ')
        contacts[u_id] = name
        print('Creating data for user: [' + u_id + '] ' + contacts[u_id])
        try:
            with open('contacts.json', 'w') as f:
                json.dump(contacts, f)
                print('wrote json file')
        except Exception as e:
            print(e)
            return

    num_imgs = int(
        input('Enter the number of pictures to take of this user: '))
    count = 0
    print('Please look at the camera')

    for frame in camera.capture_continuous(
            rawCapture, format="bgr", use_video_port=True):
        img = frame.array
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            print('Captured ' + str(count) + ' / ' + str(num_imgs) + ' images')
            cv2.imwrite(
                "dataset/User." + str(u_id) + '.' + str(count) + ".jpg",
                gray[y:y + h, x:x + w])
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= num_imgs:
            break

        rawCapture.truncate(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
