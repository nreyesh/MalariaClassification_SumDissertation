###############################################################################
# this code helps to train a sequential custom model on the dataset of your interest and
# visualize the confusion matrix, ROC and AUC curves
###############################################################################
# load the libraries
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import log_loss
from keras.optimizers import SGD
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from evaluation import plot_confusion_matrix
from sklearn.metrics import average_precision_score
from Load_data import load_data, plot_confusion_matrix
from sklearn.model_selection import train_test_split

#########################image characteristics#################################
img_rows = 100  # dimensions of image
img_cols = 100
channel = 3  # RGB
num_classes = 2
batch_size = 1
num_epoch = 60

nb_train_samples = 27558       #  training samples


#cross-validation at the patient level
train_data_dir = '/home/nicor/Documents/Summer_Project/cell_images'              #   13558 images


############################################################################################################################
"""configuring the customized model"""
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# fix the optimizer
sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)  # try varying this for your task and see the best fit

# compile the model
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
###############################################################################
# load data
X_train, Y_train, lab = load_data(train_data_dir,nb_train_samples,img_rows, img_cols)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=41)

# print the shape of the data
print(X_train.shape, Y_train.shape)
###############################################################################
# train the model
t = time.time()
hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epoch, verbose=1,
                 shuffle=True, validation_data=None)

'''
# serialize model to JSON
model_json = model.to_json()
with open("CNN_Architecture_Custom_Model.json", "w") as json_file:
    json_file.write(model_json)
# Saves model weights to HDF5
model.save_weights("CNN_Weights_Custom_Model.h5")
print("Saved model to disk")

# Saves entire model
model.save('Entire_CNN_custom_model.h5')
'''

# compute the training time
print('Training time: %s' % (time.time() - t))

###############################################################################
# predict on the validation data
print(X_test.shape, Y_test.shape)

# Make predictions
print('-' * 30)
print('Predicting on the validation data...')
print('-' * 30)
y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

# plot the confusion matrix
target_names = [ 'class 1(normal)','class 0(abnormal)']
print(classification_report(Y_test, y_pred.round(), target_names=target_names))
print(confusion_matrix(Y_test.argmax(axis=1), y_pred.round().argmax(axis=1)))
cnf_matrix = (confusion_matrix(Y_test.argmax(axis=1), y_pred.round().argmax(axis=1)))
np.set_printoptions(precision=4)
plt.figure()
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion Matrix Custom Model')
plt.show()

'''
# compute the accuracy
Test_accuracy = accuracy_score(Y_test.argmax(axis=-1), y_pred.argmax(axis=-1))
print("Test_Accuracy = ", Test_accuracy)

# compute the ROC-AUC values
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve for the positive class
plt.figure()
lw = 1  # true class label
plt.plot(fpr[1], tpr[1], color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics')
plt.legend(loc="lower right")
plt.show()

## computhe the cross-entropy loss score
score = log_loss(Y_test, y_pred)
print(score)

## compute the average precision score
prec_score = average_precision_score(Y_test, y_pred)
print(prec_score)

# transfer it back
y_pred = np.argmax(y_pred, axis=1)
Y_test = np.argmax(Y_test, axis=1)
print(y_pred)
print(Y_test)

#################compute confusion matrix######################################

# plot the confusion matrix
target_names = ['class 0(abnormal)', 'class 1(normal)']
print(classification_report(Y_test, y_pred, target_names=target_names))
print(confusion_matrix(Y_test, y_pred))
cnf_matrix = (confusion_matrix(Y_test, y_pred))
np.set_printoptions(precision=4)
plt.figure()
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
plt.show()
'''


'''
###############################################################################
# visualizing losses and accuracy
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']
xc = range(num_epoch)

plt.figure(1, figsize=(20, 10), dpi=300)
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'])
plt.style.use(['classic'])

plt.figure(2, figsize=(20, 10), dpi=300)
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'])
# plt.style.use(['classic'])
plt.style.use(['classic'])
###############################################################################
'''