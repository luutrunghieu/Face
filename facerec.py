# facerec.py
import cv2, sys, numpy, os
size = 1
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'train'

(im_width, im_height) = (112, 92)

model = cv2.face.EigenFaceRecognizer_create()
model.read("model.xml")

# Part 2: Use fisherRecognizer on camera stream
haar_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0)
while True:

    # Loop until the camera is working
    rval = False
    while(not rval):
        # Put the image from the webcam into 'frame'
        (rval, frame) = webcam.read()
        if(not rval):
            print("Failed to open webcam. Trying again...")

    # Flip the image (optional)
    frame=cv2.flip(frame,1,0)

    # Convert to grayscalel
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to speed up detection (optinal, change size above)
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    # Detect faces and loop through each one
    faces = haar_cascade.detectMultiScale(mini)
    for i in range(len(faces)):
        face_i = faces[i]

        # Coordinates of face after scaling back by `size`
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # [1]
        # Write the name of recognized face
        
        if prediction[0]==0:
            name = 'Chien'
        elif prediction[0]==1:
            name = 'Hieu'
        elif prediction[0] == 2:
            name = 'HieuVu'
        elif prediction[0] == 3:
            name = 'Thu'
        cv2.putText(frame,
           '%s - %.0f' % (name,prediction[1]),
           (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

    # Show the image and check for ESC being pressed
    cv2.imshow('OpenCV', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break