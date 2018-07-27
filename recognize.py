import cv2
import numpy as np
import os

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import time


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
    with open('contacts.json', 'r') as contacts_f:
        names = json.load(contacts_f)    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    user_times = {}
    while True:
        ret, img = cam.read()
        # img = cv2.flip(img, -1)  # Flip vertically
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
            print('id: ', id)
            print('conf: ', confidence)
            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))

                image_file = 'captured.png'
                cv2.imwrite(image_file, img)

                if id in approved:
                    if (time.time() - approved[id]) > 10:
                        send_email(image_file,"approved")
                    else:
                        # Do we want to send an email for this?
                        approved[id] = time.time()
                else:
                    approved[id] = time.time()
                    send_email(image_file,"approved")
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

                image_file = 'captured.png'
                cv2.imwrite(image_file, img)

                if len(unknowns) > 99:
                    unknowns.remove(unknowns[0])
                unknowns.append(time.time())
                if unknowns[len(unknowns)-1] > 10:
                    send_email(image_file,"unkown")

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1,
                        (255, 255, 0), 1)

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


# TODO get picture to send
def send_email(image, type):
    # Email I made for this project
    fromaddr = "eaifproj@gmail.com"
    # My wit email.
    toaddr = "kowaleskie@wit.edu"

    msg = MIMEMultipart()

    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "test"

    body = type

    msg.attach(MIMEText(body, 'plain'))

    # File location + name for adding a file
    filename = image
    attachment = open(os.getcwd()+os.sep+image, "rb")

    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    # Password for email
    server.login(fromaddr, "eaistuff")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
    return True

if __name__ == "__main__":
    main()
