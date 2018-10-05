#----------------------------------------------------------------------------------------------------#
#--------------------------------- Parasitized Cell Extraction --------------------------------------#
#----------------------------------------------------------------------------------------------------#


import numpy as np
import cv2
import os


## Image Details
img_rows = 1300
img_cols = 1029
nb_train_samples = 100
dir_img = '/home/nicor/Documents/Summer_Project/malaria-full/test'
dir_co = '/home/nicor/Documents/Summer_Project/malaria-full/labelled/Coordenates'

# Load training images
img_names = os.listdir(dir_img)
img_names.sort()
img_co = os.listdir(dir_co)
img_co.sort()

#print(img_names)
#print(img_co)

i = 0
print('-' * 30)
print('Extracting Parasatized Cells...')
print('-' * 30)

save = '/home/nicor/Documents/Summer_Project/malaria-full/Train/Parasitized/img'
save_new = '/home/nicor/Documents/Summer_Project/malaria-full/Modified_img/img'
for i in range(0,len(img_names)):
    print('Image Name: ',img_names[i])
    img = cv2.imread(os.path.join(dir_img,img_names[i]), cv2.IMREAD_COLOR)
#    cv2.imshow('image2', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

#    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

#    cv2.imshow('image2', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    print('Coordenates File Name: ',img_co[i])
    #print('Path: ',dir_co)
    point = np.loadtxt(os.path.join(dir_co, img_co[i]), delimiter=' ',dtype=np.int_)
    #print('Dimension: ',point.ndim)
    if point.ndim > 1:
        for j in range(len(point)):
            #print('Cell ',j,' of ',len(point),point[j,1],point[j,3], point[j,0],point[j,2])
            cell = img[point[j,1]-2:point[j,3]+2, point[j,0]-2:point[j,2]+2]
            #print(cell.shape)
            #cv2.imshow('Parasitized Cell',cell)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            cv2.imwrite(save+str(i)+'_'+str(j)+'.jpg', cell)

    else:
        #print('Cell ', j, ' of ', len(point), point[1], point[3], point[0], point[2])
        cell = img[point[1]-2:point[3]+2, point[0]-2:point[2]+2]

    # Puts a black patches over the infected cell
    if point.ndim > 1:
        for j in range(len(point)):
            black = np.zeros((point[j,3]-point[j,1],point[j,2]-point[j,0],3),np.uint8)
            img[point[j, 1]:point[j, 3], point[j, 0]:point[j, 2]] = black

    else:
        black = np.zeros((point[3] - point[1], point[2] - point[0], 3), np.uint8)
        img[point[1]:point[3], point[0]:point[2]] = black

    cv2.imwrite(save_new + str(i) + '.jpg', img)
 #   if i % 10 == 0:
  #      print('\n','Done: {0}/{1} images'.format(i, 100))
  #  i+= 1

    print('\n')


'''
#----------------------------------------------------------------------------------------------------#
#------------------------------- Extraction of Healthy Blood Cells ----------------------------------#
#----------------------------------------------------------------------------------------------------#

import numpy as np
import cv2
import os


# Load training images
dir_img = '/home/nicor/Documents/Summer_Project/malaria-full/Modified_img'
img_names = os.listdir(dir_img)
img_names.sort()

print('Nombres: ',img_names)

for j in range(0,49):
    img = cv2.imread(os.path.join(dir_img, img_names[j]), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #ab = cv2.resize(gray, (780, 617))
    #cv2.imshow('image1',ab)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #enhance contrast (helps makes boundaries clearer)
    clache = cv2.createCLAHE(clipLimit=60.0, tileGridSize=(8,8))
    gray = clache.apply(gray)
    #ab = cv2.resize(gray, (780, 617))
    #cv2.imshow('image2',ab)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #ab = cv2.resize(thresh, (780, 617))
    #cv2.imshow('image3',ab)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    #ab = cv2.resize(opening, (780, 617))
    #cv2.imshow('image4',ab)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    #ab = cv2.resize(sure_bg, (780, 617))
    #cv2.imshow('image5',ab)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    #ab = cv2.resize(sure_bg, (780, 617))
    #cv2.imshow('image6',ab)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]


    #ab = cv2.resize(img, (780, 617))
    #cv2.imshow('image6',ab)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    save = '/home/nicor/Documents/Summer_Project/malaria-full/Train/Uninfected/img'
    # loop over the unique segment values
    for (i, segVal) in enumerate(np.unique(markers)):
        # construct a mask for the segment
        #print("[x] inspecting segment %d" % (i))
        mask = np.zeros(img.shape[:2], dtype="uint8")
        mask[markers == segVal] = 255

        f_img = cv2.bitwise_and(img, img, mask=mask)
        # show the masked region
        #cv2.imshow("Mask", mask)
        #cv2.imshow("Applied", f_img)
        #cv2.waitKey(0)

        # Saves Image
        cv2.imwrite(save + str(j)+ '_' + str(i) + '.jpg', f_img)


'''
import cv2
import numpy as np

image = cv2.imread('/home/nicor/Documents/Summer_Project/malaria-full/Train/Parasitized/img1_3.jpg')
copy = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

edged = cv2.Canny(gray, 10, 250)
cv2.imshow('Edged', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((5, 5), np.uint8)

dilation = cv2.dilate(edged, kernel, iterations=1)
cv2.imshow('Dilation', dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

(image, cnts, hiers) = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cont = cv2.drawContours(copy, cnts, -1, (0, 0, 0), 1, cv2.LINE_AA)
cv2.imshow('Contours', cont)
cv2.waitKey(0)
cv2.destroyAllWindows()

mask = np.zeros(cont.shape[:2], dtype="uint8") * 255

# Draw the contours on the mask
cv2.drawContours(mask, cnts, -1, (255, 255, 255), -1)

# remove the contours from the image and show the resulting images
img = cv2.bitwise_and(cont, cont, mask=mask)
cv2.imshow("Mask", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 50 and h > 130:
        new_img = img[y:y + h, x:x + w]

        #cv2.imwrite('Cropped.png', new_img)
        cv2.imshow("Cropped", new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
'''


'''
import mahotas as mh
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interact, fixed


dna = mh.imread('/home/nicor/Documents/Summer_Project/malaria-full/003_NL.jpg')
print(dna.shape)
dna = dna.max(axis=2)
print(dna.shape)
plt.imshow(dna)
plt.show()

### Thresholding
T_mean = dna.mean()
print(T_mean)
plt.imshow(dna > T_mean)
plt.show()

# Reduce Noise
dnaf = mh.gaussian_filter(dna, 5.)
T_mean = dnaf.mean()
bin_image = dnaf > T_mean
plt.imshow(bin_image)
plt.show()

# Labeling Data
labeled, nr_objects = mh.label(bin_image)
print(nr_objects)

plt.imshow(labeled)
plt.jet()
plt.show()

### Separating touching cells
@interact(sigma=(1.,16.))
def check_sigma(sigma):
    dnaf = mh.gaussian_filter(dna.astype(float), sigma)
    maxima = mh.regmax(mh.stretch(dnaf))
    maxima = mh.dilate(maxima, np.ones((5,5)))
    plt.imshow(mh.as_rgb(np.maximum(255*maxima, dnaf), dnaf, dna > T_mean))
    print(maxima)
plt.show()


sigma = 10
dnaf = mh.gaussian_filter(dna.astype(float), sigma)
maxima = mh.regmax(mh.stretch(dnaf))
maxima,_= mh.label(maxima)
plt.imshow(maxima)
plt.show()

dist = mh.distance(bin_image)
plt.imshow(dist)


dist = 255 - mh.stretch(dist)
watershed = mh.cwatershed(dist, maxima)
plt.imshow(watershed)


watershed *= bin_image
plt.imshow(watershed)

# Cleaning up regions
watershed = mh.labeled.remove_bordering(watershed)
plt.imshow(watershed)
plt.show()

'''