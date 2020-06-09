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

vadim_mode = False
vadim_location = "./vadim/"

danutza_mode = True
danutza_location = "./dana/"
language = 1 # 0 - english; 1 - romanian
insults = {}

if language == 0:
    with open("insults_en.json", "r") as insultFile:
        insults = json.load(insultFile)
elif language == 1:
    with open("insults_ro.json", "r") as insultFile:
        insults = json.load(insultFile)

other_insults = insults["other"]
insult_time_delta = 20
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
        face_frame = frame[y:y+h, x:x+w]
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
    return (frame, len(frontals), [])

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
    tts = gTTS(insult, lang='en')
    try:
        os.remove(filename)
    except OSError as err:
        pass
    tts.save(filename)
    playsound(filename, False)

def load_vadim():
    global insults
    global vadim_location
    insults["vadim"] = []
    for root, dirs, files in os.walk(vadim_location):
        for name in files:
            insults["vadim"].append({"filename": vadim_location + name, "count": 0})

def load_danutza():
    global insults
    global danutza_location
    insults["danutza"] = []
    for root, dirs, files in os.walk(danutza_location):
        for name in files:
            insults["danutza"].append({"filename": danutza_location + name, "count": 0})

def vadim_tudor():
    global insults
    rand_nr = randrange(len(insults["vadim"]))
    playsound(insults["vadim"][rand_nr]["filename"], False)

def dana_budeanu():
    global insults
    rand_nr = randrange(len(insults["danutza"]))
    playsound(insults["danutza"][rand_nr]["filename"], False)

if __name__ == "__main__":

    collect_face_data = True
    face_name = "damian"

    lang_name = ""
    if language == 0:
        lang_name = "english"
    elif language == 1:
        lang_name = "romanian"
    engine.setProperty('rate', 155)
    voices = engine.getProperty('voices') 
    engine.setProperty('voice', lang_name)

    if vadim_mode:
        load_vadim()
    if danutza_mode:
        load_danutza()

    turn = 0

    while(True):
        ret, frame = cap.read()

        # Detect and recognize faces
        (processed_frame, nr_faces, faces) = detectFaceAndDrawRect(frame, 0)

        cv2.imshow('frame', processed_frame)
        c = cv2.waitKey(20)
        if (getTimeStamp() - last_insult_timestamp >= insult_time_delta) and nr_faces > 0:
            last_insult_timestamp = getTimeStamp()
            if not vadim_mode and not danutza_mode:
                insult(other_insults)
            else:
                rand_nr = randrange(1000)
                if rand_nr % 2 == 0 and vadim_mode:
                    if turn >= 2 and danutza_mode:
                        dana_budeanu()
                        turn -= 1
                        continue
                    vadim_tudor()
                    turn += 1
                else:
                    if turn <= -2 and vadim_mode:
                        vadim_tudor()
                        turn += 1
                        continue
                    dana_budeanu()
                    turn -= 1
        if c & 0xFF == ord('q'):
            break
        if c & 0xFF == ord('s') and collect_face_data:
            cv2.imwrite("capture.jpg", processed_frame)

    cap.release()
    cv2.destroyAllWindows()