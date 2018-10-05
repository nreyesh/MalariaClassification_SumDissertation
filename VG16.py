from numpy.random import seed
seed(41)

import cv2
import collections, numpy as np
import os
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from Load_data import load_data, plot_confusion_matrix

#cross-validation at the patient level
train_data_dir = '/home/nicor/Documents/Summer_Project/cell_images'              #   13558 images
#val_data_dir = '/home/nicor/Documents/Summer_Project/cell_images_2/Val'            #   4000 images

nb_train_samples = 27558       #  training samples
img_rows = 100
img_cols = 100

X,Y, lab= load_data(train_data_dir,nb_train_samples,img_rows,img_cols)
X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)


print('Original Max: ',X.shape,Y.shape)


#######        Training Model      ########

import time
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras import models
from keras import applications
from keras.optimizers import SGD
from matplotlib import pyplot as plt
channel = 3     #RGB
num_classes = 2
batch_size = 1  #vary depending on the GPU
num_epoch = 60

## ResNet-50 Model
#feature_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
#feature_model = Model(input=feature_model.input, output=feature_model.get_layer('res5c_branch2c').output)

# VGG16
feature_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))
feature_model = Model(input=feature_model.input, output=feature_model.get_layer('block5_conv2').output)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers to prevent large gradient updates wrecking the learned weights
for layer in feature_model.layers:
    layer.trainable = False

# Add the ResNet-50 convolutional base model
x = feature_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

# and a logistic layer
predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
#predictions = svm.SVC(C=7.5, kernel="linear")(x)

# Final Model
model = Model(inputs=feature_model.input, outputs=predictions)
# Getting model summary
model.summary()

#print(model.get_weights())


# compile the model (sh#ould be done *after* setting layers to non-trainable)
#fix the optimizer
sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9,  nesterov=True)
#compile the gpu model
model.compile(optimizer=sgd,
              loss='mse',
              metrics=['accuracy'])

hist = model.fit(X, Y, batch_size=batch_size, epochs=num_epoch, shuffle=True, validation_data=None,  verbose=1)

'''
# serialize model to JSON
model_json = model.to_json()
with open("VG16_weight.json", "w") as json_file:
    json_file.write(model_json)
# Saves model weights to HDF5
model.save_weights("VG16_architecture.h5")
print("Saved model to disk")

# Saves entire model
model.save('VG16_all_model.h5')
'''

#print the history of the trained model
print('History of the trained model')
print(hist.history)


# compute the training time
#print('Training time: %s' % (time.time() - t))


###############################################################################
# predict on the validation data
print(X_test.shape, Y_test.shape)
from sklearn.metrics import classification_report, confusion_matrix

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
