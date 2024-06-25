import face_recognition,time
import cv2,os,pickle
import numpy as np
from glob import glob
#import dlib,time
import mediapipe as mp

import datetime
from datetime import timedelta
#from datetime import datetime

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult

counter=0
#detector = dlib.get_frontal_face_detector()

files=[]
known_face_names=[]
known_face_encodings=[]
for file in glob('faces/*.jpg'):
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
        print(f'лицо не удалось добавить: {x}')
        #quit()
 
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(f'rtsp://login:password@192.168.1.1/RVi/1/1')

# Initialize some variables
face_locations = []
face_locations=[]

face_encodings = []
face_names = []
process_this_frame = True
counter=0

faces_entered = []
tolerance=0.5 #this only for face_recognition not face_detection. the latter has his own tolerance.

def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    global counter
        
    if len(result.detections) != 0:
        face_locations=[]
        
        #print(len(result.detections))
        image_copy = np.copy(output_image.numpy_view())

        #print(image_copy.shape)
        for detection in result.detections:
            bbox = detection.bounding_box
            #print(bbox)
            #[(88, 270, 163, 196)] top, right, bottom, left
            face_locations.append((bbox.origin_y,bbox.origin_x+bbox.width, bbox.origin_y+ bbox.height, bbox.origin_x)) #this is the right order 
            print(face_locations) #[BoundingBox(origin_x=673, origin_y=268, width=215, height=215)] #(500, 720, 509, 729)
            
##            cropped = image_copy[bbox.origin_y:bbox.origin_y + bbox.height, bbox.origin_x : bbox.origin_x + bbox.width]
##            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
##            cv2.imwrite(f'{counter}.jpg',cropped)
##            counter+=1
        try:
            face_encodings = face_recognition.face_encodings(image_copy, face_locations, model="small")
 
            for face_encoding in face_encodings:                
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                #print(name)
                           
                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]
                
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                #print(face_distances)
                #print(face_distances)

                res = any(ele <= tolerance for ele in face_distances)
                if res==True:                
                
                    #print(face_distances)
                    best_match_index = np.argmin(face_distances)
                    #print(best_match_index)
                    
                    if matches[best_match_index]:                
                        name = known_face_names[best_match_index]
                        #bbox.origin_x,bbox.origin_x + bbox.width, bbox.origin_y, bbox.origin_y + bbox.height
                        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(image_copy, (bbox.origin_x,bbox.origin_y), (bbox.origin_x+bbox.width,bbox.origin_y+ bbox.height), (0, 0, 255), 2)
                        
                        cv2.imwrite(f'{counter}.jpg',image_copy)
                        print(f'{name} {datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')
                        #quit()
                        #counter+=1
                else:
                    cv2.rectangle(image_copy, (bbox.origin_x,bbox.origin_y), (bbox.origin_x+bbox.width,bbox.origin_y+ bbox.height), (0, 0, 255), 2)
                    cv2.imwrite(f'{counter}_unknown.jpg',image_copy)
                    print(f'{name} {datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')
                    #quit()
                counter+=1
##                        
####                        if name in faces_entered:
####                            cv2.imwrite(f'{counter}.jpg',small_frame)
####                            counter+=1
####                            
####                        else:
####                            faces_entered.append(name)
####                            print(f'{name} {datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')
        except:
            pass
        

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='detector.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM, min_detection_confidence=0.7, min_suppression_threshold=0.3, result_callback=print_result) #here is confidence only for detector not for face_recognition
detector = FaceDetector.create_from_options(options)
ts=0



while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    #small_frame=frame

    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    
    # Find all the faces and face encodings in the current frame of video
    start_time = time.time()
    detector.detect_async(frame, int(ts))    
        
    #print("--- %s seconds ---" % (time.time() - start_time))

    ts+=1
    
   



