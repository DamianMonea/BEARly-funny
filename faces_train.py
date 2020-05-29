import os
import numpy as np
from PIL import Image
from PIL import ImageOps
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

frontal_face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create(neighbors = 8)

current_id = 0
label_ids = {}
y_labels = []
X_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)
            pil_img = Image.open(path).convert("L") # convert to grayscale
            size = (550, 550)
            final_img = pil_img.resize(size, Image.ANTIALIAS)
            mirrored = ImageOps.mirror(final_img)
            image_array = np.array(final_img, "uint8")
            faces = frontal_face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                X_train.append(roi)
                y_labels.append(id_)

            image_array = np.array(mirrored, "uint8")
            faces = frontal_face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                X_train.append(roi)
                y_labels.append(id_)
            


with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(X_train, np.array(y_labels))

recognizer.save("trainer.yml")