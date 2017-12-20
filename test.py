import pandas as pd
import cv2, numpy, sys, numpy, os
from sklearn.model_selection import train_test_split
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'test'

label_truth = []
label_pred = []

model = cv2.face.FisherFaceRecognizer_create()
model.read("model.xml")

for(subdirs, dirs, files) in os.walk(fn_dir):
    for subdir in dirs:
        subjectpath = os.path.join(fn_dir,subdir)
        
        for filename in os.listdir(subjectpath):
            if subdir == 'An':
                label_truth.append(0)
            elif subdir == 'Bao':
                label_truth.append(1)
            elif subdir == 'CoOanh':
                label_truth.append(2)
            elif subdir == 'Hieu':
                label_truth.append(3)
            elif subdir == 'HieuVu':
                label_truth.append(4)
            elif subdir == 'Thu':
                label_truth.append(5)
            elif subdir == 'VuAnh':
                label_truth.append(6)

            f_name, f_extension = os.path.splitext(filename)
            if(f_extension.lower() not in
                    ['.png','.jpg','.jpeg','.gif','.pgm']):
                print("Skipping "+filename+", wrong file type")
                continue
            path = subjectpath + '/' + filename
            prediction = model.predict(cv2.imread(path,0))
            # if prediction[1] >3000:
            #     # label_pred.append('?'+' - '+str(prediction[1]))
            #     label_pred.append('?')
            # else:
                # label_pred.append(str(prediction[0])+" - "+str(prediction[1]))
            label_pred.append(str(prediction[0]))
print("Predict: "+str(label_pred))
print("Truth: "+str(label_truth))
# images = []
# data = pd.read_csv('data.csv',delimiter=';',header=None)
# X = data.ix[:,0].values
# y = data.ix[:,1].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True)
# for x in X_train:
#     images.append(cv2.imread(x,0))
# (images, y_train) = [numpy.array(lis) for lis in [images, y_train]]
# model = cv2.face.EigenFaceRecognizer_create()
# model.train(images, y_train)

# predict_values = []
# for test in X_test:
#     prediction = model.predict(cv2.imread(test,0))
#     predict_values.append(prediction[0])
# sub_matrix = predict_values - y_test
# similar = 0
# for e in sub_matrix:
#     if(e == 0):
#         similar+=1
# print(similar/sub_matrix)