import cv2
import json
import os
from util import get_contacts


def main():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    face_detector = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    # For each person, enter one numeric face id
    u_id = input('Enter the id for this user: ')
    # u_id = input('\n enter user id end press <return> ==>  ')
    # Initialize individual sampling face count

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

    #if dir dataset does not exist, make it
    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    while (True):
        ret, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            print('Captured ' + str(count) + ' / ' + str(num_imgs) + ' images')
            # Save the captured image into the datasets folder
            cv2.imwrite(
                "dataset/User." + str(u_id) + '.' + str(count) + ".jpg",
                gray[y:y + h, x:x + w])
            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= num_imgs:  # Take 30 face sample and stop video
            break
    # Do a bit of cleanup
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
