import numpy as np
import matplotlib.pyplot as plt
from Load_data import load_data, plot_confusion_matrix, load_data_2
from Feature_extractor import ObtainFeatures
from keras.models import load_model
from sklearn.metrics import accuracy_score, average_precision_score, classification_report
from sklearn.metrics import confusion_matrix,log_loss,f1_score
import cv2 as cv

## Loads Data
img_rows = 100  # dimensions of image
img_cols = 100
nb_test_samples = 7935  ## Erode_base # training samples
#nb_test_samples = 7910  ## Erode_base_2.0 # training samples

train_data_dir = '/home/nicor/Documents/Summer_Project/malaria-full/Train/Erode_base/Neutral'

# Loads Data
X_CNN, Y_CNN,label = load_data(train_data_dir,nb_test_samples,img_rows, img_cols)

label = [s.replace('.jpg', '') for s in label]
print(label[0])

for i in range(0,5):
    print(Y_CNN[i,0],Y_CNN[i,1])

res_cnn_1 = '/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Neutral/'
res_cnn_2 = '/home/nicor/Documents/Summer_Project/Results/Images/CNN/Erode_base/Neutral/'
res_svm =  '/home/nicor/Documents/Summer_Project/Results/Images/SVM/Erode_base/Neutral/'

##------------- VGG-16 RESULTS -------------##
'''
model = load_model('/home/nicor/PycharmProjects/Summer_Project/VG16_all_model.h5')
y_pred = model.predict(X_CNN, batch_size=1, verbose=1)
y_pred = y_pred.round()

for i in range(0,nb_test_samples):
    if y_pred[i,0] == 1:
        if Y_CNN[i,0] == 1:
            cv.imwrite(res_cnn_1 + label[i] + '_normal_' + 'pred_normal.jpg', X_CNN[i]*255)
        else:
            cv.imwrite(res_cnn_1 + label[i] + '_abnormal_' + 'pred_normal.jpg', X_CNN[i]*255)
    else:
        if Y_CNN[i, 0] == 1:
            cv.imwrite(res_cnn_1 + label[i] + '_normal_' + 'pred_abnormal.jpg', X_CNN[i] * 255)
        else:
            cv.imwrite(res_cnn_1 + label[i] + '_abnormal_' + 'pred_abnormal.jpg', X_CNN[i] * 255)

Test_accuracy = accuracy_score(Y_CNN.argmax(axis=-1), y_pred.round().argmax(axis=-1))
print("Test_Accuracy = ", Test_accuracy)

## computhe the cross-entropy loss score
score = log_loss(Y_CNN, y_pred)
print(score)

## compute the average precision score
prec_score = average_precision_score(Y_CNN, y_pred.round())
print(prec_score)

f1 = f1_score(Y_CNN, y_pred.round(), average=None)
print('F1_score: ',f1)


# plot the confusion matrix
target_names = [ 'class 1(normal)','class 0(abnormal)']
print(classification_report(Y_CNN, y_pred.round(), target_names=target_names))
print(confusion_matrix(Y_CNN.argmax(axis=1), y_pred.round().argmax(axis=1)))
cnf_matrix = (confusion_matrix(Y_CNN.argmax(axis=1), y_pred.round().argmax(axis=1)))
np.set_printoptions(precision=4)
plt.figure()
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion Matrix VGG-16')
plt.show()


##------------- CUSTOM MODEL RESULTS -------------##

# Loads Custom Model
model = load_model('/home/nicor/PycharmProjects/Summer_Project/Entire_CNN_custom_model.h5')
y_pred = model.predict(X_CNN, batch_size=1, verbose=1)
y_pred = y_pred.round()

for i in range(0,nb_test_samples):
    if y_pred[i,0] == 1:
        if Y_CNN[i,0] == 1:
            cv.imwrite(res_cnn_2 + label[i] + '_normal_' + 'pred_normal.jpg', X_CNN[i]*255)
        else:
            cv.imwrite(res_cnn_2 + label[i] + '_abnormal_' + 'pred_normal.jpg', X_CNN[i]*255)
    else:
        if Y_CNN[i, 0] == 1:
            cv.imwrite(res_cnn_2 + label[i] + '_normal_' + 'pred_abnormal.jpg', X_CNN[i] * 255)
        else:
            cv.imwrite(res_cnn_2 + label[i] + '_abnormal_' + 'pred_abnormal.jpg', X_CNN[i] * 255)

Test_accuracy = accuracy_score(Y_CNN.argmax(axis=-1), y_pred.round().argmax(axis=-1))
print("Test_Accuracy = ", Test_accuracy)

## computhe the cross-entropy loss score
score = log_loss(Y_CNN, y_pred)
print(score)

## compute the average precision score
prec_score = average_precision_score(Y_CNN, y_pred.round())
print(prec_score)

f1 = f1_score(Y_CNN, y_pred.round(), average=None)
print('F1_score: ',f1)


# plot the confusion matrix
target_names = [ 'class 1(normal)','class 0(abnormal)']
print(classification_report(Y_CNN, y_pred.round(), target_names=target_names))
print(confusion_matrix(Y_CNN.argmax(axis=1), y_pred.round().argmax(axis=1)))
cnf_matrix = (confusion_matrix(Y_CNN.argmax(axis=1), y_pred.round().argmax(axis=1)))
np.set_printoptions(precision=4)
plt.figure()
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion Matrix Custom Model')
plt.show()

'''
##------------- SUPPORT VECTOR MACHIME MODEL RESULTS -------------##
from sklearn.externals import joblib

# load the model from disk
filename = 'SVM_linear_final.sav'
model = joblib.load(filename)

# Load data
X_svm, Y_svm, label_svm = load_data_2(train_data_dir,nb_test_samples,img_rows, img_cols)
label_svm = [s.replace('.jpg', '') for s in label_svm]
#print(label_svm[0])
#cv.imshow('Image Resized', X_svm[0])
#cv.waitKey(0)

X_svm2 = ObtainFeatures(X_svm,len(X_svm))
y_pred = model.predict(X_svm2)

for i in range(0,nb_test_samples):
    if y_pred[i] == 0:
        if Y_svm[i] == 0:
            cv.imwrite(res_svm + label_svm[i] + '_normal_' + 'pred_normal.png', X_svm[i])
        else:
            cv.imwrite(res_svm + label_svm[i] + '_abnormal_' + 'pred_normal.png', X_svm[i])
    else:
        if Y_svm[i] == 0:
            cv.imwrite(res_svm + label_svm[i] + '_normal_' + 'pred_abnormal.png', X_svm[i])
        else:
            cv.imwrite(res_svm + label_svm[i] + '_abnormal_' + 'pred_abnormal.png', X_svm[i])

Test_accuracy = accuracy_score(Y_svm, y_pred)
print("Test_Accuracy = ", Test_accuracy)

## computhe the cross-entropy loss score
score = log_loss(Y_svm, y_pred)
print(score)

## compute the average precision score
prec_score = average_precision_score(Y_svm, y_pred)
print(prec_score)

f1 = f1_score(Y_svm, y_pred, average=None)
print('F1_score: ',f1)


# plot the confusion matrix
target_names = [ 'class 1(normal)','class 0(abnormal)']
print(classification_report(Y_svm, y_pred, target_names=target_names))
print(confusion_matrix(Y_svm, y_pred))
cnf_matrix = (confusion_matrix(Y_svm, y_pred))
np.set_printoptions(precision=4)
plt.figure()
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion Matrix SVM')
plt.show()

