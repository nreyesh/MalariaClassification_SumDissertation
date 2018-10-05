
import cv2
import numpy as np
import os
from keras.utils import np_utils
import matplotlib.pyplot as plt
import itertools


## Loads data for CNNs and normalizes pixels
def load_data(train_data_dir,nb_train_samples,img_rows_orig,img_cols_orig):
    # Load training images
    labels = os.listdir(train_data_dir)
    total = len(labels)

    X = np.ndarray((nb_train_samples, img_rows_orig, img_cols_orig, 3), dtype=np.float32)
    Y = np.zeros((nb_train_samples,), dtype='uint8')

    print('Data type X',X.dtype)
    print('Data type Y',Y.dtype,'\n')

    lab = []
    i = 0
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    for label in labels:
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        total = len(image_names_train)
        print(label, total)
        if label == 'Parasitized':
            j = 1
        else:
            j = 0

        for image_name in image_names_train:
            lab.append(image_name)
            img = cv2.imread(os.path.join(train_data_dir, label, image_name), cv2.IMREAD_COLOR)
            img2 = cv2.resize(img,(img_rows_orig,img_cols_orig))
#            print('Norm Pixels', img[50, 52, 0], img[50, 52, 1], img[50, 52, 2])
#            cv2.imshow('Image Resized', img2)
#            cv2.waitKey(0)

            img3 = img2 / 255
#            print('Norm Pixels', img3[50, 52, 0], img3[50, 52, 1], img3[50, 52, 2])
#            cv2.imshow('Image Normalized', img3)
#            cv2.waitKey(0)

            X[i] = img3
            Y[i] = j
#            print('Final Pixels', X[i, 50, 52, 0], X[i, 50, 52, 0], X[i, 50, 52, 0])
#            cv2.imshow('Image Stored', X[i])
#            cv2.waitKey(0)

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
    print(i)



    print('Transform targets to keras compatible format.')
    Y = np_utils.to_categorical(Y[:nb_train_samples], 2)

    return X, Y,lab


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,  # if true all values in confusion matrix is between 0 and 1
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


## Loads data for alternative classifiers
def load_data_2(train_data_dir,nb_train_samples,img_rows_orig,img_cols_orig):
    # Load training images
    labels = os.listdir(train_data_dir)
    total = len(labels)

    X = np.ndarray((nb_train_samples, img_rows_orig, img_cols_orig, 3), dtype=np.uint8)
    Y = np.zeros((nb_train_samples,), dtype='uint8')

    print('Data type X',X.dtype)
    print('Data type Y',Y.dtype,'\n')

    lab = []
    i = 0
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    for label in labels:
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        total = len(image_names_train)
        print(label, total)
        if label == 'Parasitized':
            j = 1
        else:
            j = 0

        for image_name in image_names_train:
            lab.append(image_name)
            img = cv2.imread(os.path.join(train_data_dir, label, image_name), cv2.IMREAD_COLOR)
            img2 = cv2.resize(img,(img_rows_orig,img_cols_orig))
#            print('Norm Pixels', img[50, 52, 0], img[50, 52, 1], img[50, 52, 2])
#            cv2.imshow('Image Resized', img2)
#            cv2.waitKey(0)

            X[i] = img2
            Y[i] = j
#            print('Final Pixels', X[i, 50, 52, 0], X[i, 50, 52, 0], X[i, 50, 52, 0])
#            cv2.imshow('Image Stored', X[i])
#            cv2.waitKey(0)

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
    print(i)

    return X, Y, lab


def threshold_slow(T, image):
    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            image[y, x] = 0 if image[y, x] != T else 1

    # return the thresholded image
    return image