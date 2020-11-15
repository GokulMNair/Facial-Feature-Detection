# Draw broders on face and all facial features using dlib facelandmark 68 points

import cv2
import dlib
from scipy.spatial import distance


#Start the camera to Detect face
cap = cv2.VideoCapture(0)

# dlib - A toolkit for making real world machine learning and data analysis applications
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("/home/pi/Documents/Drowsy/shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Uisng dlib's hog_face_detector to detect face from the input image
    faces = hog_face_detector(gray)
    for face in faces:
        # Using dlib_facelandmark function get the landmark points on the gray image like for left eye it is 36 to 41
        face_landmarks = dlib_facelandmark(gray,face)
        
        # For the jaw border of face, ranges from 1 to 17
        for n in range(0,16):
            # for the landmark at point value 'n' get the x and y coordinates
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            next_point = n + 1            
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
        
        # From the points of left eye, the points range from 37 to 42
        for n in range(36,42):
            # for the landmark at point value 'n' get the x and y coordinates
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
            
        # From the points of left eyebrow, the points range from 18 to 22
        for n in range(17,21):
            # for the landmark at point value 'n' get the x and y coordinates
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            next_point = n + 1            
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
            
        
        # From the points of right eyebrow, the points range from 18 to 22
        for n in range(23,26):
            # for the landmark at point value 'n' get the x and y coordinates
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            next_point = n + 1            
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)       
         
         # From the points of right eye, the points range from 42 to 47
        for n in range(42,48):
            # for the landmark at point value 'n' get the x and y coordinates
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
         
        # From the points of mouth, the points range from 61 to 68
        for n in range(60,68):
            # for the landmark at point value 'n' get the x and y coordinates
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            
            next_point = n + 1
            if n == 67:
                next_point = 60                       
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
            
        # From the points of outer mouth, the points range from 49 to 60
        for n in range(48,60):
            # for the landmark at point value 'n' get the x and y coordinates
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            
            next_point = n + 1
            if n == 59:
                next_point = 48                       
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
            
        
        # From the points of nose, the points range from 28 to 31
        for n in range(27,30):
            # for the landmark at point value 'n' get the x and y coordinates
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            next_point = n + 1            
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
            
        # From the points of nose, the points range from 32 to 36
        for n in range(31,35):
            # for the landmark at point value 'n' get the x and y coordinates
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            next_point = n + 1            
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

    cv2.imshow('camera',frame) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
   
    
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()

