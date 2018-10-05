
import numpy as np
from matplotlib import pyplot as plt
import cv2
from Load_data import load_data_2
import os
from scipy.stats import entropy

img_h = cv2.imread('/home/nicor/Documents/Summer_Project/noninfected.png', cv2.IMREAD_COLOR)
img_nh = cv2.imread('/home/nicor/Documents/Summer_Project/infected.png', cv2.IMREAD_COLOR)


img_rows = 100  # dimensions of image
img_cols = 100
channel = 3  # RGB
nb_train_samples =  240

train_data_dir = '/home/nicor/Documents/Summer_Project/Sample'
train_data_dir_2 = '/home/nicor/Documents/Summer_Project/Sample_3/Parasitized'

#X, Y,lab = load_data_2(train_data_dir,nb_train_samples,img_rows, img_cols)
#X_test, Y_test ,lab_2 = load_data_2(train_data_dir,nb_train_samples,img_rows, img_cols)
labels = os.listdir(train_data_dir)
#print(X.shape)

## GrayScale Histogram
'''
hist_h = []
hist_nh = []


for label in labels:
    image_names_train = os.listdir(os.path.join(train_data_dir, label))
    total = len(image_names_train)
    print(label, total)
    for image_name in image_names_train:
        img = cv2.imread(os.path.join(train_data_dir, label, image_name), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (100, 100))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if label == 'Uninfected':
            hist_h.append(cv2.calcHist([rgb], [0], None, [128], [0, 256]))
        else:
            hist_nh.append(cv2.calcHist([rgb], [0], None, [128], [0, 256]))

hist_h = np.array(hist_h)
hist_nh = np.array(hist_nh)

he = np.zeros(128)
nh = np.zeros(128)

for i in range(0,128):
    he[i] = sum(hist_h[:,i])/120
    nh[i] = sum(hist_nh[:,i])/120

hist_nh_pred_n = []
hist_nh_pred_nh = []
labels = os.listdir(train_data_dir_2)
print(labels)

for label in labels:
    image_names_train = os.listdir(os.path.join(train_data_dir_2, label))
    total = len(image_names_train)
    print(label, total)
    for image_name in image_names_train:
        img = cv2.imread(os.path.join(train_data_dir_2, label, image_name), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (100, 100))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if label == 'abnormal':
            hist_nh_pred_nh.append(cv2.calcHist([rgb], [0], None, [128], [0, 256]))
        else:
            hist_nh_pred_n.append(cv2.calcHist([rgb], [0], None, [128], [0, 256]))

hist_nh_pred_nh = np.array(hist_nh_pred_nh)
hist_nh_pred_n = np.array(hist_nh_pred_n)

nh_pred_n = np.zeros(128)
nh_pred_nh = np.zeros(128)

for i in range(0,128):
    nh_pred_n[i] = sum(hist_nh_pred_n[:,i])/10
    nh_pred_nh[i] = sum(hist_nh_pred_nh[:,i])/10
'''


'''
print('GrayScale')
print('Healthy: ',entropy(hey))
print('Non Healthy: ',entropy(nhx),'\n')
'''


'''
f, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(hey,color='b')
ax[1].plot(nhx,color='r')

ax[0].set(ylabel='Uninfected Cell')
ax[1].set(ylabel='Parasitized Cell')
ax[1].set(xlabel='Number of Bins')
f.text(0, 0.5, 'Number of Pixels', va='center', rotation='vertical')
plt.show()
'''

'''
f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
ax[0,0].plot(nh,color='r')
ax[0,1].plot(he,color='b')
ax[0,0].set(ylabel='Train Set')
ax[0,0].set_title('Parasitized')
ax[0,1].set_title('Uninfected')

ax[1,0].plot(nh_pred_nh,color='r',ls='--',label='')
ax[1,1].plot(nh_pred_n,color='r',ls='--',label='')
ax[1,0].set(ylabel='Parasitized Cell Test Set')
ax[1,0].set_title('Correctly Classified')
ax[1,1].set_title('Misclassified')

f.text(0, 0.5, 'Number of Pixels', va='center', rotation='vertical')
f.text(0.4, 0.03, 'Number of Bins', va='center')
plt.show()
'''

## Saturation Level Histogram
'''
hist_h =[]
hist_nh =[]

for label in labels:
    image_names_train = os.listdir(os.path.join(train_data_dir, label))
    total = len(image_names_train)
    print(label, total)
    for image_name in image_names_train:
        img = cv2.imread(os.path.join(train_data_dir, label, image_name), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if label == 'Healthy':
            hist_h.append(cv2.calcHist([rgb], [1], None, [256], [0, 256]))
        else:
            hist_nh.append(cv2.calcHist([rgb], [1], None, [256], [0, 256]))

hist_h = np.array(hist_h)
hist_nh = np.array(hist_nh)


hey = np.zeros(256)
nhx = np.zeros(256)

for i in range(0,128):
    hey[i] = sum(hist_h[:,i])/120
    nhx[i] = sum(hist_nh[:,i])/120

print('Saturation')
print('Healthy: ',entropy(hey))
print('Non Healthy: ',entropy(nhx),'\n')
'''

'''
f, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(hey,color='b')
ax[1].plot(nhx,color='r')
ax[0].set(ylabel='Uninfected Cell')
ax[1].set(ylabel='Parasitized Cell')
ax[1].set(xlabel='Number of Bins')
f.text(0, 0.5, 'Number of Pixels', va='center', rotation='vertical')
plt.show()



plt.plot(hey,label='Uninfected Cell',color='b')
plt.plot(nhx,label='Parasitized Cell',color='r')
plt.xlabel('Number of bins')
plt.ylabel('Number of Pixels')
plt.legend(loc='upper right')
plt.show()
'''

'''


hsv_h = cv2.cvtColor(img_h, cv2.COLOR_BGR2HSV)
hsv_nh = cv2.cvtColor(img_nh, cv2.COLOR_BGR2HSV)
hist_h = cv2.calcHist([hsv_h], [1], None, [256], [0, 256])
hist_nh = cv2.calcHist([hsv_nh], [1], None, [256], [0, 256])

plt.plot(hist_h,label='Uninfected Cell',color='b')
plt.plot(hist_nh,label='Parasitized Cell',color='r')
plt.xlabel('Number of bins')
plt.ylabel('Number of Pixels')
plt.legend(loc='upper right')
plt.show()
'''

## RGB Normalized Histogram

hist_h1 =[]
hist_h2 =[]
hist_h3 =[]
hist_nh1 =[]
hist_nh2 =[]
hist_nh3 =[]

for label in labels:
    image_names_train = os.listdir(os.path.join(train_data_dir, label))
    total = len(image_names_train)
    print(label, total)
    for image_name in image_names_train:
        img = cv2.imread(os.path.join(train_data_dir, label, image_name), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (100, 100))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if label == 'Uninfected':
            hist_h1.append(cv2.calcHist([rgb], [0], None, [256], [0, 256]))
            hist_h2.append(cv2.calcHist([rgb], [1], None, [256], [0, 256]))
            hist_h3.append(cv2.calcHist([rgb], [2], None, [256], [0, 256]))

        else:
            hist_nh1.append(cv2.calcHist([rgb], [0], None, [256], [0, 256]))
            hist_nh2.append(cv2.calcHist([rgb], [1], None, [256], [0, 256]))
            hist_nh3.append(cv2.calcHist([rgb], [2], None, [256], [0, 256]))


hist_h1 = np.array(hist_h1)
hist_h2 = np.array(hist_h2)
hist_h3 = np.array(hist_h3)
hist_nh1 = np.array(hist_nh1)
hist_nh2 = np.array(hist_nh2)
hist_nh3 = np.array(hist_nh3)

red_h = np.zeros(256)
green_h = np.zeros(256)
blue_h = np.zeros(256)
red_nh = np.zeros(256)
green_nh = np.zeros(256)
blue_nh = np.zeros(256)


for i in range(0,256):
    red_h[i] = sum(hist_h1[:,i])/120
    green_h[i] = sum(hist_h2[:,i])/120
    blue_h[i] = sum(hist_h3[:,i])/120
    red_nh[i] = sum(hist_nh1[:,i])/120
    green_nh[i] = sum(hist_nh2[:,i])/120
    blue_nh[i] = sum(hist_nh3[:,i])/120

'''
print('Color')
print('Healthy: ',entropy(red_h))
print('Non Healthy: ',entropy(red_nh))
print('Healthy: ',entropy(green_h))
print('Non Healthy: ',entropy(green_nh),'\n')

print(red_h[0])
print(green_h[0])
print(blue_h[0])

f, ax = plt.subplots(2, 3, sharex=True, sharey=True)
ax[0,0].plot(red_h,color='r')
ax[0,1].plot(green_h,color='g')
ax[0,2].plot(blue_h,color='b')
ax[1,0].plot(red_nh,color='r',ls='--')
ax[1,1].plot(green_nh,color='g',ls='--')
ax[1,2].plot(blue_nh,color='b',ls='--')

ax[0,0].set(ylabel='Uninfected Cell')
ax[1,0].set(ylabel='Parasitized Cell')
ax[1,1].set(xlabel='Number of Bins')
f.text(0, 0.5, 'Number of Pixels', va='center', rotation='vertical')

ax[0, 0].set_title('Red Channel')
ax[0, 1].set_title('Green Channel')
ax[0, 2].set_title('Blue Channel')

plt.show()
'''


hist_nh_pred_nh1 =[]
hist_nh_pred_nh2 =[]
hist_nh_pred_nh3 =[]
hist_nh_pred_n1 =[]
hist_nh_pred_n2 =[]
hist_nh_pred_n3 =[]
labels2 = os.listdir(train_data_dir_2)

for label in labels2:
    image_names_train = os.listdir(os.path.join(train_data_dir_2, label))
    total = len(image_names_train)
    print(label, total)
    for image_name in image_names_train:
        img = cv2.imread(os.path.join(train_data_dir_2, label, image_name), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (100, 100))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if label == 'abnormal':
            hist_nh_pred_nh1.append(cv2.calcHist([rgb], [0], None, [256], [0, 256]))
            hist_nh_pred_nh2.append(cv2.calcHist([rgb], [1], None, [256], [0, 256]))
            #hist_nh_pred_nh3.append(cv2.calcHist([rgb], [2], None, [256], [0, 256]))

        else:
            hist_nh_pred_n1.append(cv2.calcHist([rgb], [0], None, [256], [0, 256]))
            hist_nh_pred_n2.append(cv2.calcHist([rgb], [1], None, [256], [0, 256]))
            #hist_nh_pred_n3.append(cv2.calcHist([rgb], [2], None, [256], [0, 256]))



hist_nh_pred_nh1 = np.array(hist_nh_pred_nh1)
hist_nh_pred_nh2 = np.array(hist_nh_pred_nh2)
#hist_nh_pred_nh3 = np.array(hist_nh_pred_nh3)
hist_nh_pred_n1 = np.array(hist_nh_pred_n1)
hist_nh_pred_n2 = np.array(hist_nh_pred_n2)
#hist_nh_pred_n3 = np.array(hist_nh_pred_n3)

red_nh_pred_nh = np.zeros(256)
green_nh_pred_nh = np.zeros(256)
red_nh_pred_n = np.zeros(256)
green_nh_pred_n = np.zeros(256)



for i in range(0,256):
    red_nh_pred_nh[i] = sum(hist_nh_pred_nh1[:,i])/10
    green_nh_pred_nh[i] = sum(hist_nh_pred_nh2[:,i])/10
    red_nh_pred_n[i] = sum(hist_nh_pred_n1[:,i])/10
    green_nh_pred_n[i] = sum(hist_nh_pred_n2[:,i])/10


f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
ax[0,0].plot(red_nh,color='r')
ax[0,1].plot(red_h,color='b')
ax[0,0].set_title('Parasitized')
ax[0,1].set_title('Uninfected')
ax[0,0].set(ylabel='Train set')

ax[1,0].plot(red_nh_pred_nh,color='r',ls='--')
ax[1,1].plot(red_nh_pred_n,color='r',ls='--')
ax[1,0].set_title('Correctly Classified')
ax[1,1].set_title('Misclassified')
ax[1,0].set(ylabel='Parasitized Cell Test set')

f.text(0, 0.5, 'Number of Pixels', va='center', rotation='vertical')
f.text(0.5, 0.03, 'Number of bins', va='center')
plt.show()


f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
ax[0,0].plot(green_nh,color='r')
ax[0,1].plot(green_h,color='b')
ax[0,0].set_title('Parasitized')
ax[0,1].set_title('Uninfected')
ax[0,0].set(ylabel='Train set')

ax[1,0].plot(green_nh_pred_nh,color='r',ls='--')
ax[1,1].plot(green_nh_pred_n,color='r',ls='--')
ax[1,0].set_title('Correctly Classified')
ax[1,1].set_title('Misclassified')
ax[1,0].set(ylabel='Parasitized Cell Test set')

f.text(0, 0.5, 'Number of Pixels', va='center', rotation='vertical')
f.text(0.5, 0.03, 'Number of bins', va='center')
plt.show()


## Sobel Histogram
'''
hist_hx =[]
hist_hy = []
hist_nhx =[]
hist_nhy =[]

for label in labels:
    image_names_train = os.listdir(os.path.join(train_data_dir, label))
    total = len(image_names_train)
    print(label, total)
    for image_name in image_names_train:
        img = cv2.imread(os.path.join(train_data_dir, label, image_name), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobx = cv2.Sobel(rgb, cv2.CV_32F, 1, 0, ksize=3)
        soby = cv2.Sobel(rgb, cv2.CV_32F, 0, 1, ksize=3)

        if label == 'Healthy':
            hist_hx.append(cv2.calcHist([sobx], [0], None, [60], [0, 256]))
            hist_hy.append(cv2.calcHist([soby], [0], None, [60], [0, 256]))

        else:
            hist_nhx.append(cv2.calcHist([sobx], [0], None, [60], [0, 256]))
            hist_nhy.append(cv2.calcHist([soby], [0], None, [60], [0, 256]))

hist_hx = np.array(hist_hx)
hist_hy = np.array(hist_hy)
hist_nhx = np.array(hist_nhx)
hist_nhy = np.array(hist_nhy)

hex = np.zeros(60)
hey = np.zeros(60)
nhx = np.zeros(60)
nhy = np.zeros(60)

for i in range(0,60):
    hex[i] = sum(hist_hx[:,i]) /120
    hey[i] = sum(hist_hy[:,i]) / 120
    nhx[i] = sum(hist_nhx[:,i]) /120
    nhy[i] = sum(hist_nhy[:,i]) /120


f, ax = plt.subplots(1, 2, sharex=True,sharey=True)
ax[0].plot(hex,color='b',label='Uninfected')
ax[1].plot(hey,color='b',label='Uninfected')
ax[0].plot(nhx,color='r',label='Parasitized')
ax[1].plot(nhy,color='r',label='Parasitized')
ax[1].legend(loc="upper right")
ax[0].legend(loc="upper right")
ax[0].set_title('Sobel X-axis')
ax[1].set_title('Sobel Y-axis')
f.text(0.4, 0.02, 'Number of Bins', va='center')
f.text(0.02, 0.5, 'Number of Pixels', va='center', rotation='vertical')
plt.show()
'''

'''
gray_h = cv2.cvtColor(img_h, cv2.COLOR_BGR2GRAY)
gray_nh = cv2.cvtColor(img_nh, cv2.COLOR_BGR2GRAY)

lap_h = cv2.Laplacian(gray_h, cv2.CV_32F)
lap_nh = cv2.Laplacian(gray_nh, cv2.CV_32F)

hist_h = cv2.calcHist([lap_h], [0], None, [64], [0, 256])
hist_nh = cv2.calcHist([lap_nh], [0], None, [64], [0, 256])

plt.plot(hist_h,label='Uninfected Cell',color='b')
plt.plot(hist_nh,label='Parasitized Cell',color='r')
plt.xlabel('Number of bins')
plt.ylabel('Number of Pixels')
plt.legend(loc='upper right')
plt.show()
'''


## Contrast - Tamura Feature
'''
hist_h =[]
hist_nh =[]

for label in labels:
    image_names_train = os.listdir(os.path.join(train_data_dir, label))
    total = len(image_names_train)
    print(label, total)
    for image_name in image_names_train:
        img = cv2.imread(os.path.join(train_data_dir, label, image_name), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        if label == 'Healthy':
            hist_h.append(cv2.calcHist([gray], [0], None, [64], [0, 256]))
        else:
            hist_nh.append(cv2.calcHist([gray], [0], None, [64], [0, 256]))

hist_h = np.array(hist_h)
hist_nh = np.array(hist_nh)

hey = np.zeros(64)
nhx = np.zeros(64)

for i in range(0,64):
    hey[i] = sum(hist_h[:,i])/120
    nhx[i] = sum(hist_nh[:,i])/120

print(hey)
print(nhx)
'''

'''
f, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(hey,color='b')
ax[1].plot(nhx,color='r')
ax[0].set(ylabel='Uninfected Cell')
ax[1].set(ylabel='Parasitized Cell')
ax[1].set(xlabel='Number of Bins')
f.text(0, 0.5, 'Number of Pixels', va='center', rotation='vertical')
plt.show()


gray_h = cv2.cvtColor(img_h, cv2.COLOR_BGR2GRAY)
gray_nh = cv2.cvtColor(img_nh, cv2.COLOR_BGR2GRAY)

gray_h = cv2.equalizeHist(gray_h)
gray_nh = cv2.equalizeHist(gray_nh)

hist_h = cv2.calcHist([gray_h], [0], None, [64], [0, 256])
hist_nh = cv2.calcHist([gray_nh], [0], None, [64], [0, 256])

print(entropy(hist_h))
print(entropy(hist_nh))
'''
'''
plt.plot(hey,label='Uninfected Cell',color='b')
plt.plot(nhx,label='Parasitized Cell',color='r')
plt.xlabel('Number of bins')
plt.ylabel('Number of Pixels')
plt.legend(loc='upper right')
plt.show()
'''

'''
sobelx = cv2.Sobel(img_h,cv2.CV_32F,1,0,ksize=3)
sobely = cv2.Sobel(img_h,cv2.CV_32F,0,1,ksize=3)

cv2.imshow('img',sobelx)
cv2.waitKey(0)

cv2.imshow('img',sobely)
cv2.waitKey(0)

sobelx = cv2.Sobel(img_nh,cv2.CV_32F,1,0,ksize=3)
sobely = cv2.Sobel(img_nh,cv2.CV_32F,0,1,ksize=3)

cv2.imshow('img',sobelx)
cv2.waitKey(0)

cv2.imshow('img',sobely)
cv2.waitKey(0)
'''

'''
median = cv2.bilateralFilter(image,5,15,15)
gray_bl = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
'''

'''
plt.hist(gray.ravel(), 256, [0, 256])
plt.show()

ret3,th3 = cv2.threshold(gray_b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ab = cv2.resize(th3, (780, 617))
cv2.imshow('Threshold',ab)
cv2.waitKey(0)

gaus2 = cv2.adaptiveThreshold(gray_bl,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,181,2) #701 #71 la raja invertido
ab = cv2.resize(gaus2, (780, 617))
cv2.imshow('Adaptive Threshold',ab)
cv2.waitKey(0)

fig, axs = plt.subplots(1,2, sharex=True,sharey=True)

axs[0].imshow(th3)
axs[1].imshow(gaus2)

axs[0].set_title('Otsu Threshold')
axs[1].set_title('Adaptive Threshold')
axs[0].set_yticklabels([])
axs[1].set_xticklabels([])
fig.tight_layout()
plt.show()
'''
'''
gaus1 = cv2.adaptiveThreshold(gray_bf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,2) #701  #71 la raja invertido
ab = cv2.resize(gaus1, (780, 617))
cv2.imshow('Block Size 51',ab)
cv2.waitKey(0)

gaus2 = cv2.adaptiveThreshold(gray_bf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,2) #701  #71 la raja invertido
ab = cv2.resize(gaus2, (780, 617))
cv2.imshow('Block Size 101',ab)
cv2.waitKey(0)

gaus3 = cv2.adaptiveThreshold(gray_bf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,151,2) #701 #71 la raja invertido
ab = cv2.resize(gaus3, (780, 617))
cv2.imshow('Block Size 151',ab)
cv2.waitKey(0)

gaus4 = cv2.adaptiveThreshold(gray_bf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,201,2) #701 #71 la raja invertido
ab = cv2.resize(gaus4, (780, 617))
cv2.imshow('Block Size 201',ab)
cv2.waitKey(0)

gaus5 = cv2.adaptiveThreshold(gray_bf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,301,2) #701 #71 la raja invertido
ab = cv2.resize(gaus5, (780, 617))
cv2.imshow('Block Size 301',ab)
cv2.waitKey(0)

gaus6 = cv2.adaptiveThreshold(gray_bf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,501,2) #701 #71 la raja invertido
ab = cv2.resize(gaus6, (780, 617))
cv2.imshow('Block Size 501',ab)
cv2.waitKey(0)
'''

'''

image = cv2.imread('/home/nicor/Documents/Summer_Project/malaria-full/test/001.jpg')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.blur(image,(7,7))
gray_bf = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

gaus3 = cv2.adaptiveThreshold(gray_bf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,151,2) #701 #71 la raja invertido
ab = cv2.resize(gaus3, (780, 617))
cv2.imshow('Block Size 151',ab)
cv2.waitKey(0)
cv2.destroyAllWindows()



## Removes small white dots
# find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(gaus3, connectivity=4)
# connectedComponentswithStats yields every seperated component with information on each of them, such as size
# the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1];
nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
# here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 1500

# your answer image
img2 = np.zeros(output.shape)
# for every component in the image, you keep it only if it's above min_size
for j in range(0, nb_components):
    if sizes[j] >= min_size:
        img2[output == j + 1] = 255

ab = cv2.resize(img2, (780, 617))
cv2.imshow('White Dots Removed', ab)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_inv = (255 - np.round(img2))
img_inv = img_inv.astype('uint8')
ab = cv2.resize(img_inv, (780, 617))
cv2.imshow('Inverted Image', ab)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((15,15), np.uint8)
opening = cv2.morphologyEx(img_inv, cv2.MORPH_OPEN, kernel)
ab = cv2.resize(opening, (780, 617))
cv2.imshow('Opening', ab)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((2, 2), np.uint8)
eroded = cv2.erode(opening, kernel, iterations=3)
ab = cv2.resize(eroded, (780, 617))
cv2.imshow('Erosion', ab)
cv2.waitKey(0)
cv2.destroyAllWindows()


nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=4)
# connectedComponentswithStats yields every seperated component with information on each of them, such as size
# the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1];
nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
# here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 1000

# your answer image
img_inv2 = np.zeros(output.shape)
# for every component in the image, you keep it only if it's above min_size
for j in range(0, nb_components):
    if sizes[j] >= min_size:
        img_inv2[output == j + 1] = 255

img_inv2 = img_inv2.astype('uint8')
ab = cv2.resize(img_inv2, (780, 617))
cv2.imshow('White Dots Removed',ab)
cv2.waitKey(0)
cv2.destroyAllWindows()

np.savetxt('/home/nicor/Documents/Summer_Project/binary_image.out', img_inv2, delimiter=',')

fig, axs = plt.subplots(3,2, sharex=True,sharey=True)

axs[0,0].imshow(gaus3)
axs[0,1].imshow(img2)
axs[1,0].imshow(img_inv)
axs[1,1].imshow(opening)
axs[2,0].imshow(eroded)
axs[2,1].imshow(img_inv2)

axs[0,0].set_title('Binary Image')
axs[0,1].set_title('White Areas Removed')
axs[1,0].set_title('Inverted Image')
axs[1,1].set_title('Opening')
axs[2,0].set_title('Erotion')
axs[2,1].set_title('Impurities Removed')
axs[0,0].set_yticklabels([])
axs[1,0].set_yticklabels([])
axs[1,0].set_xticklabels([])
fig.tight_layout()
plt.show()
'''


'''
fig, axs = plt.subplots(1,2, sharex=True,sharey=True)

axs[0].imshow(gaus3)
axs[1].imshow(gaus4)

axs[0].set_title('Block Size = 151')
axs[1].set_title('Block Size = 201')
axs[0].set_yticklabels([])
axs[0].set_xticklabels([])
fig.tight_layout()
plt.show()
'''


'''
#### STEP 1

from keras.models import load_model
from keras import applications
from keras import backend as K

# build the VGG16 network
model = load_model('/home/nicor/PycharmProjects/Summer_Project/VG16_all_model.h5')
model.summary()

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

model = load_model('/home/nicor/PycharmProjects/Summer_Project/Entire_CNN_custom_model.h5')
model.summary()

'''

'''
norm1 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Eroade_bas/Black/8_74_abnormal_pred_normal.jpg', cv2.IMREAD_COLOR)
norm2 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Eroade_bas/Black/7_163_abnormal_pred_normal.jpg', cv2.IMREAD_COLOR)
norm3 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Eroade_bas/Black/19_60_abnormal_pred_normal.jpg', cv2.IMREAD_COLOR)
img1 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Eroade_bas/Black/9_14_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Eroade_bas/Black/0_90_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
img3 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Eroade_bas/Black/6_71_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)

fig, axs = plt.subplots(2,3, sharex=True,sharey=True)

axs[0,0].imshow(norm1)
axs[1,0].imshow(img1)
axs[0,1].imshow(norm2)
axs[1,1].imshow(img2)
axs[0,2].imshow(norm3)
axs[1,2].imshow(img3)

#axs[0,0].set_title('Misclassified')
#axs[0,1].set_title('Corrctly Classified')
axs[0,0].set(ylabel='Misclassified')
axs[1,0].set(ylabel='Correctly Classified')
axs[0,0].set_yticklabels([])
axs[1,0].set_yticklabels([])
axs[1,0].set_xticklabels([])
fig.tight_layout()
plt.show()
'''

### Compare cells classified with Black and White Background
'''
dir1 = '/home/nicor/Documents/Summer_Project/Results/Images/CNN/Eroade_bas/Black'
dir2 = '/home/nicor/Documents/Summer_Project/Results/Images/CNN/Eroade_bas/Neutral'

image_names1 = os.listdir(dir1)
image_names2 = os.listdir(dir2)

image_names1 = [s.replace('.jpg', '') for s in image_names1]
image_names2 = [s.replace('.jpg', '') for s in image_names2]

#image_names1 = \
image_names1.sort()
#image_names2 = \
image_names2.sort()


print('Sort name images')
print(len(image_names1),image_names1)
print(len(image_names2),image_names2)

img_name = [image_names1, image_names2]
print(len(img_name),img_name)

img_name = list(map(list, zip(*img_name)))

file = open('/home/nicor/Documents/classification_result_CNN.txt', 'w')
for item in img_name:
  file.write("%s\n" % item)

'''


### VGG16 Parasitized class
'''
norm1 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/5_93_abnormal_pred_normal.jpg', cv2.IMREAD_COLOR)
norm2 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/22_132_abnormal_pred_normal.jpg', cv2.IMREAD_COLOR)
norm3 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/43_136_abnormal_pred_normal.jpg', cv2.IMREAD_COLOR)
img1 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/2_39_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/22_5_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
img3 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/10_68_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)

fig, axs = plt.subplots(2,3, sharex=True,sharey=True)

axs[0,0].imshow(norm1)
axs[1,0].imshow(img1)
axs[0,1].imshow(norm2)
axs[1,1].imshow(img2)
axs[0,2].imshow(norm3)
axs[1,2].imshow(img3)

#axs[0,0].set_title('Misclassified')
#axs[0,1].set_title('Corrctly Classified')
axs[0,0].set(ylabel='Misclassified')
axs[1,0].set(ylabel='Correctly Classified')
axs[0,0].set_yticklabels([])
axs[1,0].set_yticklabels([])
axs[1,0].set_xticklabels([])
fig.tight_layout()
plt.show()
'''

'''
### VGG16 Parasitized class
norm1 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/5_93_abnormal_pred_normal.jpg', cv2.IMREAD_COLOR)
norm2 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/22_132_abnormal_pred_normal.jpg', cv2.IMREAD_COLOR)
norm3 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/43_136_abnormal_pred_normal.jpg', cv2.IMREAD_COLOR)
img1 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/2_39_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/22_5_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
img3 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/10_68_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)

fig, axs = plt.subplots(2,3, sharex=True,sharey=True)

axs[0,0].imshow(norm1)
axs[1,0].imshow(img1)
axs[0,1].imshow(norm2)
axs[1,1].imshow(img2)
axs[0,2].imshow(norm3)
axs[1,2].imshow(img3)

#axs[0,0].set_title('Misclassified')
#axs[0,1].set_title('Corrctly Classified')
axs[0,0].set(ylabel='Misclassified')
axs[1,0].set(ylabel='Correctly Classified')
axs[0,0].set_yticklabels([])
axs[1,0].set_yticklabels([])
axs[1,0].set_xticklabels([])
fig.tight_layout()
plt.show()
'''


### Compare classification among different models Mc Nemar test
'''
dir1 = '/home/nicor/Documents/Summer_Project/Results/Images/CNN/Erode_base/Black'
dir2 = '/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black'
dir3 = '/home/nicor/Documents/Summer_Project/Results/Images/SVM/Erode_base/Black'

image_names1 = os.listdir(dir1)
image_names2 = os.listdir(dir2)
image_names3 = os.listdir(dir3)

image_names1 = [s.replace('.jpg', '') for s in image_names1]
image_names2 = [s.replace('.jpg', '') for s in image_names2]
image_names3 = [s.replace('.png', '') for s in image_names3]

#image_names1 = \
image_names1.sort()
#image_names2 = \
image_names2.sort()

image_names3.sort()

print('Sort name images')
print(len(image_names1),image_names1)
print(len(image_names2),image_names2)
print(len(image_names3),image_names3)

img_name = [image_names1, image_names2, image_names3]
print(len(img_name),img_name)

img_name = list(map(list, zip(*img_name)))

file = open('/home/nicor/Documents/classification_result_per_cell.txt', 'w')
for item in img_name:
  file.write("%s\n" % item)
'''

## Black and White CNN
'''
dir1 = '/home/nicor/Documents/Summer_Project/Results/Images/CNN/Erode_base/Black'
dir2 = '/home/nicor/Documents/Summer_Project/Results/Images/CNN/Erode_base/Neutral'

image_names1 = os.listdir(dir1)
image_names2 = os.listdir(dir2)

image_names1 = [s.replace('.jpg', '') for s in image_names1]
image_names2 = [s.replace('.jpg', '') for s in image_names2]


#image_names1 = \
image_names1.sort()
#image_names2 = \
image_names2.sort()


print('Sort name images')
print(len(image_names1),image_names1)
print(len(image_names2),image_names2)

img_name = [image_names1, image_names2]
print(len(img_name),img_name)

img_name = list(map(list, zip(*img_name)))

file = open('/home/nicor/Documents/CNN_white_black_background.txt', 'w')
for item in img_name:
  file.write("%s\n" % item)
'''

'''
## CNN vs VGG
dir1 = '/home/nicor/Documents/Summer_Project/Results/Images/CNN/Erode_base/Black'
dir2 = '/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black'

image_names1 = os.listdir(dir1)
image_names2 = os.listdir(dir2)

image_names1 = [s.replace('.jpg', '') for s in image_names1]
image_names2 = [s.replace('.jpg', '') for s in image_names2]


#image_names1 = \
image_names1.sort()
#image_names2 = \
image_names2.sort()


print('Sort name images')
print(len(image_names1),image_names1)
print(len(image_names2),image_names2)

img_name = [image_names1, image_names2]
print(len(img_name),img_name)

img_name = list(map(list, zip(*img_name)))

file = open('/home/nicor/Documents/CNN_VGG_black_background.txt', 'w')
for item in img_name:
  file.write("%s\n" % item)
'''

## VGG vs CNN parasitized class
'''
vgg1 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/18_108_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
vgg2 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/18_161_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
vgg3 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/18_46_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
cnn1 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Erode_base/Black/10_127_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
cnn2 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Erode_base/Black/10_116_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
cnn3 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Erode_base/Black/10_35_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)


fig, axs = plt.subplots(3,2, sharex=True,sharey=True)

axs[0,0].imshow(vgg1)
axs[0,1].imshow(cnn1)
axs[1,0].imshow(vgg2)
axs[1,1].imshow(cnn2)
axs[2,0].imshow(vgg3)
axs[2,1].imshow(cnn3)

axs[0,0].set_title('CNN')
axs[0,1].set_title('VGG-16')
fig.text(0.1, 0.5, 'Misclassified by VGG', va='center', rotation='vertical')
fig.text(0.9, 0.5, 'Misclassified by CNN', va='center', rotation='vertical')
axs[0,0].set_yticklabels([])
axs[1,0].set_yticklabels([])
axs[1,0].set_xticklabels([])
fig.tight_layout()
plt.show()



vgg1 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/43_144_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
vgg2 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/43_27_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
vgg3 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/43_83_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
cnn1 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Erode_base/Black/41_51_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
cnn2 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Erode_base/Black/41_56_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
cnn3 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Erode_base/Black/41_81_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)


fig, axs = plt.subplots(3,2, sharex=True,sharey=True)

axs[0,0].imshow(vgg1)
axs[0,1].imshow(cnn1)
axs[1,0].imshow(vgg2)
axs[1,1].imshow(cnn2)
axs[2,0].imshow(vgg3)
axs[2,1].imshow(cnn3)

axs[0,0].set_title('CNN')
axs[0,1].set_title('VGG-16')
fig.text(0.1, 0.5, 'Misclassified by VGG', va='center', rotation='vertical')
fig.text(0.9, 0.5, 'Misclassified by CNN', va='center', rotation='vertical')
axs[0,0].set_yticklabels([])
axs[1,0].set_yticklabels([])
axs[1,0].set_xticklabels([])
fig.tight_layout()
plt.show()
'''


'''
vgg1 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/1_41_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
vgg2 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/10_107_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
vgg3 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/VGG/Erode_base/Black/12_60_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
cnn1 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Erode_base/Black/13_145_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
cnn2 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Erode_base/Black/14_70_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)
cnn3 = cv2.imread('/home/nicor/Documents/Summer_Project/Results/Images/CNN/Erode_base/Black/15_39_abnormal_pred_abnormal.jpg', cv2.IMREAD_COLOR)


fig, axs = plt.subplots(3,2, sharex=True,sharey=True)

axs[0,0].imshow(vgg1)
axs[0,1].imshow(cnn1)
axs[1,0].imshow(vgg2)
axs[1,1].imshow(cnn2)
axs[2,0].imshow(vgg3)
axs[2,1].imshow(cnn3)

#axs[0,0].set_title('CNN')
#axs[0,1].set_title('VGG-16')
#fig.text(0.1, 0.5, 'Misclassified by VGG', va='center', rotation='vertical')
#fig.text(0.9, 0.5, 'Misclassified by CNN', va='center', rotation='vertical')
axs[0,0].set_yticklabels([])
axs[1,0].set_yticklabels([])
axs[1,0].set_xticklabels([])
fig.tight_layout()
plt.show()
'''