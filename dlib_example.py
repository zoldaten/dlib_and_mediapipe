import face_recognition
import cv2,os,pickle
import numpy as np
from glob import glob
import dlib,time

import datetime
from datetime import timedelta
#from datetime import datetime

counter=0
detector = dlib.get_frontal_face_detector()

files=[]
known_face_names=[]
known_face_encodings=[]
for file in glob('faces/*.jpg'): #faces - dir for faces to work with
    print(file)
    files.append(file)
    known_face_names.append(file.split('\\')[1].split('.')[0])
    
print (known_face_names)
files_new=[face_recognition.load_image_file(x) for x in files]
for x,y in zip(known_face_names,files_new):
    #print(x)
    try:
        known_face_encodings.append((face_recognition.face_encodings(y))[0])
    except Exception as e:
        print(f'face add failed: {x}')
        #quit()

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(f'rtsp://login:password@192.168.1.1:554/RVi/1/1')


# Initialize some variables
face_locations = []
face_locations=[]
#face_locations=[eval(x.split(' \n')[0]) for x in open('faces.txt').readlines()]

face_encodings = []
face_names = []
process_this_frame = True
counter=0

faces_entered = []

tolerance=0.6 #less is more strict

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    #small_frame=frame
     
    # Only process every other frame of video to save time
    if process_this_frame:
      # Resize frame of video to 1/4 size for faster face recognition processing
      small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) # 0.03sec
  
      #print(small_frame.shape)
  
      # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
      #rgb_small_frame = small_frame[:, :, ::-1]
      rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
      
      # Find all the faces and face encodings in the current frame of video
      start_time = time.time()
      
      face_locations = face_recognition.face_locations(rgb_small_frame)
     
      if len(face_locations) != 0:
          print(face_locations)
      
      print("--- %s seconds ---" % (time.time() - start_time))
         
      face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model="small")
       
      for face_encoding in face_encodings:
          # See if the face is a match for the known face(s)
          matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
          name = "Unknown"
          
                     
          # # If a match was found in known_face_encodings, just use the first one.
          # if True in matches:
          #     first_match_index = matches.index(True)
          #     name = known_face_names[first_match_index]
  
          face_distances = face_recognition.face_distance(known_face_encodings, face_encoding) 
          #print(face_distances)
  
          res = any(ele <= tolerance for ele in face_distances)
          if res==True:
          
          
              #print(face_distances)
              best_match_index = np.argmin(face_distances)
              #print(best_match_index) 22
              
              if matches[best_match_index]:                
                  name = known_face_names[best_match_index]
                  cv2.imwrite(f'{counter}.jpg',small_frame)
                  print(f'{name} {datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')
                  counter+=1
              else:
                  cv2.imwrite(f'{counter}_unknown.jpg',small_frame)
                  print(f'{name} {datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')
                  counter+=1
         

    process_this_frame = not process_this_frame
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()



