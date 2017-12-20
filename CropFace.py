import cv2, sys, numpy, os
size = 10
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'test'
fn_out = 'train2'
haar_cascade = cv2.CascadeClassifier(fn_haar)
(im_width, im_height) = (112, 92)
for (subdirs,dirs, files) in os.walk(fn_dir):
    for subdir in dirs:
        subjectpath = os.path.join(fn_dir,subdir)
        subpath = os.path.join(fn_out,subdir)
        if not os.path.isdir(subpath):
            os.mkdir(subpath)
        pin=sorted([int(n[:n.find('.')]) for n in os.listdir(subpath)
        if n[0]!='.' ]+[0])[-1] + 1
        for filename in os.listdir(subjectpath):
            path = subjectpath + '\\' + filename
            print(path)
            image = cv2.imread(path)
            height, width, channels = image.shape
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
            faces = haar_cascade.detectMultiScale(mini)
            faces = sorted(faces, key=lambda x: x[3])
            if faces:
                face_i = faces[0]
                (x, y, w, h) = [v * size for v in face_i]

                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (im_width, im_height))
                cv2.imwrite('%s/%s.png' % (subpath, pin), face_resize)
                pin += 1