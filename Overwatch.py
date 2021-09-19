# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 12:03:52 2021

@author: Administrator
"""

import os
import requests

import face_recognition
import cv2
import datetime

# logic of Overwatch for our group prototype
# first of all, open the camera and read the photo captured by the camera，
# Locate the face of the customer in the screen 
# Then frame the face with a color frame
# 2. read the names and facial features of customer in the face database
# 3. Using images of customer's faces to match facial features in a face database,
# The customer's name is marked above the blue box of the customer's profile picture
# and the unverified users are marked as unkown customer
# 4. positioning and locking the target, change the use of relevabt color frame to target the face of the frame

# Webhook URL links to notification. When Overwatch detects face, the phone will get a notification from IFTTT Mobile APP.
headers = {'user-agent':'Mozilla/5.0 Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36'}
event = 'Phone'
url = 'https://maker.ifttt.com/trigger/' + event + '/with/key/j9_Kv-Bn2AqIvCZvCRStYi3gvS1kLeDIOJjbwx0KtVe'
param = {'value1':"Warning, the number of people is more than 3, please keep 1.5m distance of each person!"}

#create two more two sequences to store the name of vip visitor and verfied name 
vip_names = ['VipVisitor1', 'VipVisitor2','VipVisitor3']
Verified_name = ['VerifiedVisitor1','VerifiedVisitor2','VerifiedVisitor3']

#MAIn process 
face_databases_dir = 'face_databases' #connect the face database with our system
customer_names = [] # save customer names
customer_faces_encodings = [] # Save customer facial feature vector (one-to-one correspondence)
#  Get all file names in face_databases_dir folder
files = os.listdir('face_databases')
#  Loop through the file name for further processing
for image_shot_name in files:
    # The first part is stored as the customer name in the user_names list
    customer_name, _ = os.path.splitext(image_shot_name)
    customer_names.append(customer_name)

    # The facial feature information from the image file is stored in user_faces_encodings
    image_file_name = os.path.join(face_databases_dir, image_shot_name)
    image_file = face_recognition.load_image_file(image_file_name)
    face_shot_encoding = face_recognition.face_encodings(image_file)[0]
    customer_faces_encodings.append(face_shot_encoding)
# Open the camera and obtain the camera object
video_capture = cv2.VideoCapture(0) # 0 represents the first camera-0
# Loop over and over again to capture the footage from the camera and do further processing

while True:
    # Obtaining images captured by the camera
    ret, frame = video_capture.read() # frame picture taken by a camera
    # The state of the system will be displayed in the upper left corner of screen
    cv2.putText(frame, "System is operating", (10, 20),
                cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 0, 0), 2)
    # The datetime will be displayed in the left bottom of screen
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX,
                0.55, (0, 255, 0), 2)
    # Extract the area of the customer's face from the photo (there may be more than one)
    face_locations = face_recognition.face_locations(frame)
    # Extracting facial features from the area of all customer's faces
    # (there may be more than one)
    # [' facial features corresponding to the first face ', 'facial features corresponding to the second face'...]
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    # Define a list for storing the names of customer who was captured
    # [' First customer's name ', 'second customer's name'...]
    # If the facial feature does not match the feature in the database, the value is Unknown
    CustomerList = []
    # traversing face_encodings，match the facial features to the previous database
    for face_shot_encoding in face_encodings:
        # compare_faces([' Facial features 1', 'Facial features 2',' Facial features 3'...] Unknown facial features)
        # compare_faces return result
        # Suppose the unknown facial feature matches facial feature 1
        # And facial feature 2 doesn't match facial feature 3
        
        # Suppose the unknown facial feature matches facial feature 2
        # But does not match facial feature 1 and facial feature 3
        matchs = face_recognition.compare_faces(customer_faces_encodings, face_shot_encoding)
        # [' First customer's name ', 'second customer's name', 'third customer's name'...]
        visitor_name = "UnKnown Visitor"
        for index, is_match in enumerate(matchs): 
            if is_match:
                visitor_name = customer_names[index]
                break
        CustomerList.append(visitor_name)
    
    # if number of people is more than 3, there is a notification send to the phone
    if len(CustomerList)>=3:    
        requests.get(url, data=param, headers=headers)
    # Cycle through the area where the customer's face is
    # Draw a frame, and mark the customer's name on the frame
    # zip([' first customer's position ', 'second customer's position '], [' first customer's name ',' second customer's name '])
    for (top, right, bottom, left), visitor_name in zip(face_locations, CustomerList):
        
        color = (255, 0, 0)# the default color is red
        if visitor_name in vip_names:# if the face was matched as the vip_name, then the color will be green
            color = (0, 255, 0)
        elif visitor_name in Verified_name:# if the face was matched as the Verified_name, then the color will be green
            color = (0, 0, 255)
        # Paint the frame in the area where the face is located
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, visitor_name, (left, top-10), font, 0.5, color, 2)
    
    # Display the shot and framed pictures through OpencV
    cv2.imshow("Overwatch", frame)
    # Set a mechanism for exiting the While loop by pressing f
    if cv2.waitKey(1) & 0xFF == ord('f'):
        break # Exit the while loop
# When exiting the program, release the camera or other resources
video_capture.release()
cv2.destroyAllWindows()