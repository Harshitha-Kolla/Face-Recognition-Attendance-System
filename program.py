#ATTENDANCE RECOGNITION SYSTEM BY HARSHITHA AND LARANYA

import face_recognition 
import cv2
import numpy as np
import csv
import os
from datetime import datetime

vid_cap = cv2.VideoCapture(0)

steve_img = face_recognition.load_image_file("pics/steve_jobs.jpg")
steve_face_encoding = face_recognition.face_encodings(steve_img)[0]

elon_img = face_recognition.load_image_file("pics/elon_musk.jpg")
elon_face_encoding = face_recognition.face_encodings(elon_img)[0]

bill_img = face_recognition.load_image_file("pics/bill_gates.jpg")
bill_face_encoding = face_recognition.face_encodings(bill_img)[0]

known_face_encodings = [
    steve_face_encoding, 
    elon_face_encoding,
    bill_face_encoding
]

known_face_names = [
    "Steve Jobs",
    "Elon Musk",
    "Bill Gates"
]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

now = datetime.now()
date = now.strftime("%d-%m-%Y")

f = open(date+'.csv', 'w+',newline='')
writer = csv.writer(f)

while True:
    ret, frame = vid_cap.read()
    frame = cv2.imread("pics/steve_jobs.jpg")
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_dist = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_dist)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
    
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    writer.writerow([name, now.strftime("%H-%M-%S")])
                    print(name, "marked present")

    cv2.imshow('attendance system', frame)

    #exit program by pressing 'q' on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid_cap.release()
cv2.destroyAllWindows()
f.close()   