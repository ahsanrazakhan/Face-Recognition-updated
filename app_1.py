import cv2
import os
from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import json

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

import os, sys
import pickle
import face_recognition
import cvzone
import math
from PIL import Image
import random
import time

from flask import Flask, render_template, Response, request, jsonify
import re
import h5py

import pickle

import mysql.connector
from flask import send_from_directory
from flask import jsonify

# Defining Flask App
app = Flask(__name__)


def db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234abcd",
        database="student_auth_db"
    )
    return conn


def get_student_record(id):
    '''
    take student id as input

    returns
    ID
    Student image associated with ID
    Student info associated with Image
    '''
    connection = db_connection()
    cursor = connection.cursor()

    # fetch the information
    cursor.execute("SELECT id, name, age, degree, image_path FROM students WHERE id = %s", (id,))
    result = cursor.fetchone()

    # Provide the image path
    image_directory = r'C:\Program Files\MySQL\MySQL Server 8.0\student_image'

    if result:
        student = {
            "id": result[0],
            "name": result[1],
            "age": result[2],
            "degree": result[3],
            "image_path": result[4]
        }
    else:
        connection.close()
        return None, None, None

    # Fetching student image from file system
    image_path = os.path.join(image_directory, student["image_path"])
    
    image_path = os.path.abspath(image_path)

    if os.path.exists(image_path):
        imgStudent = cv2.imread(image_path)
    else:
        print(f"Image file not found: {image_path}")
        connection.close()
        return None, None, None

    connection.close()

    return id, imgStudent, student


def recognize_faces(student_id):

    # Here are some of the modes of operations: 0 processing, 1 matched 2 not matched
    modeType = 0

    counter = 0

    id = -1

    imgStudent = []

    # call the student function to get the id and related image

    id, imgStudent, studentInfo = get_student_record(id=student_id)
    #print(imgStudent['name'])
    #print(img)
    #print(id)

    if imgStudent is None:
        print("Student image not found.")
        return

    # ... (other parts of the code remain the same)


    # Generate the known face encoding for comparision
    known_face_encoding = face_recognition.face_encodings(imgStudent)[0]

    cap = cv2.VideoCapture(0)
    #if not cap.isOpened():
    #    sys.exit('Video Source not Found')

    face_match = []

    start_time = None
    start_time = time.time()

    # face detection loop



    while(time.time() - start_time) < 5:
        ret, img = cap.read()
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        #imgBackground[162:162+480, 55:55+640] = img
        #imgBackground[44:44+633, 808:808+414] = imgModeList[modeType]

        #imgBackground= img
        #imgBackground[44:44+633, 808:808+414] = imgModeList[modeType]

        for face_encoding, faceLoc in zip(face_encodings, face_locations):
                # See if the face is a match for the student record
            match = face_recognition.compare_faces([known_face_encoding], face_encoding, tolerance=0.50)[0]

            face_distances = face_recognition.face_distance([known_face_encoding], face_encoding)
                
            if match:
                face_match.append(1)
            else:
                face_match.append(0)

                #name = id

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4       # because the size os 1/4
            bbox = x1, y1, x2-x1, y2-y1

                # draw bounding box around the face
                #cv2.rectangle(img, bbox, (0, 0, 255), 2)

            img= cvzone.cornerRect(img, bbox)
                #imgStudent = cv2.resize(imgStudent, (216, 216), interpolation=cv2.INTER_LINEAR)
            
            #print(match)
            #print(face_distances)

            #cv2.imshow('frame', img)

            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

            ret, jpeg = cv2.imencode('.jpg', img)
            img = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


    
    
    cap.release()
    cv2.destroyAllWindows()
    with h5py.File('temp.h5', 'w') as f:
        f.create_dataset('list', data=face_match)
    
    #return face_match

def face_check(lst):
    # call the facial recognition function here to get the decision
    
    print(lst)

    ones = lst.count(1)
    fifty_per = len(lst) * 0.5

    if ones >= fifty_per:
        return "yes"
    else:
        return "no"


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    #names,rolls,times,l = extract_attendance()  
    home = render_template('home.html') 
    #import json
    #app_url = '127.0.0.1:5000'

    #print('---------------------Home-----------------------Page---------')
    #print(home) 
    return home

# ####  App Route to start revognition and accept the data for outside

current_student_id = None

@app.route('/start_recognition', methods=['GET', 'POST'])
def start_recognition():
    global current_student_id
    if request.method == 'POST':
        data = request.json
        student_id = data.get('student_id')
    else:
        student_id = request.args.get('student_id', type=int)

    if student_id:
        current_student_id = student_id   # set studentID as global veriable to use in other routes
        return Response(recognize_faces(student_id), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({"error": "Student ID not provided"}), 400



#### Start of the facial recognition
#@app.route('/start')
#def start(): 
   # return Response(recognize_faces(student_id), mimetype='multipart/x-mixed-replace; boundary=frame')
    #recognize_faces()
    #return render_template('home.html')



@app.route('/get_app_url')
def app_url():
    url = 'http://127.0.0.1:5000'

    url_dict = {'url': url}

    url_json = json.dumps(url_dict)

    #print(url_json)
    return url_json

@app.route('/results_page')
def results_page():
    return render_template('results_page.html')



@app.route('/results', methods=['POST'])
def results():
    global current_student_id
    with h5py.File('temp.h5', 'r') as f:
        lst = list(f['list'])

    status = face_check(lst)

    if status == 'yes':
        id, imgStudent, studentInfo = get_student_record(current_student_id)
        print('Known User')
        user_info = {
            "id": id,
            "name": studentInfo['name']
        }
        return jsonify(status='yes', user_info=user_info)
    else:
        print('Unknown User')
        return jsonify(status='no')

@app.route('/status')
def status():
    global current_student_id
    with h5py.File('temp.h5', 'r') as f:
        lst = list(f['list'])

    status = face_check(lst)

    if status == 'yes':
        id, imgStudent, studentInfo = get_student_record(current_student_id)
        #print('Known User')
        user_info = {
            "id": id,
            "name": studentInfo['name']
        }
        return jsonify(status='yes', user_info=user_info)
    else:
        return jsonify(status='Not Authenticated')

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)