import cv2,sys, numpy, os
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'train'

print('Training...')

(images, labels, id) = ([], [], 0)

for(subdirs, dirs, files) in os.walk(fn_dir):
    for subdir in dirs:
        subjectpath = os.path.join(fn_dir,subdir)
        if subdir == 'An':
            id = 0
        elif subdir == 'Bao':
            id = 1
        elif subdir == 'CoOanh':
            id = 2
        elif subdir == 'Hieu':
            id = 3
        elif subdir == 'HieuVu':
            id = 4
        elif subdir == 'Thu':
            id = 5
        elif subdir == 'VuAnh':
            id = 6
        for filename in os.listdir(subjectpath):
            f_name, f_extension = os.path.splitext(filename)
            if(f_extension.lower() not in
                    ['.png','.jpg','.jpeg','.gif','.pgm']):
                print("Skipping "+filename+", wrong file type")
                continue
            path = subjectpath + '/' + filename
            label = id

            images.append(cv2.imread(path,0))
            labels.append(int(label))
(im_width, im_height) = (112, 92)
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

model = cv2.face.EigenFaceRecognizer_create()
model.train(images, labels)

model.write("model.xml")

print("Done")