import cv2
size = 10
imagePath = 'data/An/20.JPG'
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
image = cv2.imread(imagePath)
image = cv2.resize(image, (int(image.shape[1] /size ), int(image.shape[0] / size)))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# equ = cv2.equalizeHist(gray)
faces = faceCascade.detectMultiScale(
    gray
)
print ("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow("Faces found" ,image)
cv2.waitKey(0)