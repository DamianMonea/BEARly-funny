import numpy as np
import cv2
import pickle
import json
import os
from playsound import playsound
from gtts import gTTS
import pyttsx3
from random import randrange
from datetime import datetime

frontal_face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
profile_face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')

# recognizer = cv2.face.LBPHFaceRecognizer_create(neighbors = 8)

vadim_mode = True
vadim_location = "./vadim/"
vadim_files = ["ordine.mp3", "oglinda.mp3"]
language = 1 # 0 - english; 1 - romanian
insults = {}
if language == 0:
    with open("insults_en.json", "r") as insultFile:
        insults = json.load(insultFile)
elif language == 1:
    with open("insults_ro.json", "r") as insultFile:
        insults = json.load(insultFile)

thanks1 = "Am văzut că v-a plăcut ursulețul cu insulte, deci acest tik tok este prezentat de el."
thanks2 = " Vă multumesc pentru 20 de mii de urmăritori. Mă bucur că încă vă plac videoclipurile pe care le postez."
thanks3 = " Sunteti cei mai tari!"
other_insults = insults["other"]
insult_time_delta = 8
TTFI = int(insult_time_delta * 0.75) # Time to first insult
last_insult_timestamp = int(datetime.now().timestamp()) - TTFI
smallest_insult_count = 0

engine = pyttsx3.init()

labels = {}
with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}

try:
    cap = cv2.VideoCapture(0)
except cv2.error as e:
    exit

def getTimeStamp():
    return int(datetime.now().timestamp())

def mergeDetections(frontal, profile):
    faces = []
    for (xf, yf, wf, hf) in frontal:
        if len(profile) == 0:
            faces.append(((xf, yf, wf, hf)))
        for (xp, yp, wp, hp) in profile:
            f_x_ctr = xf + (wf / 2)
            f_y_ctr = yf + (hf / 2)
            p_x_ctr = xp + (wp / 2)
            p_y_ctr = yp + (hp / 2)
            dist = sqrt((f_x_ctr - p_x_ctr)**2 + (f_y_ctr - p_y_ctr)**2)
            if (dist < 15):
                faces.append((xf, yf, wf, hf))
    return faces

def detectFaceAndDrawRect(frame, recognizer):
    global frontal_face_cascade
    global profile_face_cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frontals = frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # profiles = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # faces = mergeDetections(frontals, profiles)
    for (x, y, w, h) in frontals:
        roi_gray = gray[y:y+h, x:x+w]
        roi = frame[y:y+h, x:x+w]

        #recognize faces in ROI
        # id_, conf = recognizer.predict(roi_gray)
        # if conf >= 45:
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     name = labels[id_]
        #     color = (255, 255, 255)
        #     stroke = 2
        #     cv2.putText(frame, name, (x,y - 5), font, 1, color, stroke, cv2.LINE_AA)

        color = (255, 0, 0) #BGR, not RGB
        stroke = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
    return (frame, len(frontals))

def getRandRange(insultSet):
    global smallest_insult_count
    res = 0
    for insult in insultSet:
        if insult["count"] == smallest_insult_count:
            res += 1
    if res == 0:
        smallest_insult_count += 1
        for insult in insultSet:
            if insult["count"] == smallest_insult_count:
                res += 1
        return res
    return res

def insult(insultSet):
    idx = randrange(getRandRange(insultSet))
    current_idx = 0
    for elem in insultSet:
        if elem["count"] == smallest_insult_count:
            current_idx += 1
        if (current_idx == idx):
            break
    current_idx = (len(insultSet) - 1) if current_idx >= len(insultSet) else current_idx
    insult = insultSet[current_idx]["insult"]
    insultSet[current_idx]["count"] += 1
    filename = "temp.mp3"
    # tts = gTTS(insult, lang='en')
    # tts.save(filename)
    # playsound(filename)
    # os.remove(filename)
    engine.say(thanks1)
    engine.runAndWait()
    engine.say(thanks2)
    engine.runAndWait()
    engine.say(thanks3)
    engine.runAndWait()

aux = 0
def pitica_nenorocita():
    global aux
    #idx = randrange(int((len(vadim_files) * 20) % len(vadim_files)) + 1)
    idx = aux
    aux += 1
    if (idx < 2):
        selected_file = vadim_location + vadim_files[idx]
        playsound(selected_file)

if __name__ == "__main__":

    lang_name = ""
    if language == 0:
        lang_name = "english"
    elif language == 1:
        lang_name = "romanian"
    engine.setProperty('rate', 155)
    voices = engine.getProperty('voices') 
    engine.setProperty('voice', lang_name)

    while(True):
        ret, frame = cap.read()

        # Detect and recognize faces
        (processed_frame, nr_faces) = detectFaceAndDrawRect(frame, 0)

        cv2.imshow('frame', processed_frame)
        c = cv2.waitKey(20)
        if (getTimeStamp() - last_insult_timestamp >= insult_time_delta) and nr_faces > 0:
            last_insult_timestamp = getTimeStamp()
            if not vadim_mode:
                insult(other_insults)
            else:
                pitica_nenorocita()
        if c & 0xFF == ord('q'):
            break
        if c & 0xFF == ord('s'):
            cv2.imwrite("capture.jpg", processed_frame)

    cap.release()
    cv2.destroyAllWindows()