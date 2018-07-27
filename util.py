import os
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


def get_contacts():
    contacts = {}
    try:
        with open('contacts.json', 'r') as contacts_f:
            contacts = json.load(contacts_f)
    except FileNotFoundError as e:
        raise e
    except json.JSONDecodeError as e:
        raise e
    except Exception as e:
        raise e

    return contacts


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
    attachment = open(os.getcwd() + os.sep + image, "rb")

    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition',
                    "attachment; filename= %s" % filename)

    msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    # Password for email
    server.login(fromaddr, "eaistuff")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
    return True