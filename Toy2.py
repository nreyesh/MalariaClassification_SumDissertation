
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#img = cv.imread('/home/nicor/Documents/Summer_Project/malaria-full/Train/Parasitized/img1_3.jpg')
img = cv.imread('/home/nicor/Documents/Summer_Project/malaria-full/Train/Parasitized/img0_0.jpg')
img = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray,(5,5),0)
ab = cv.resize(gray, (780, 617))
cv.imshow('Grey Image',ab)
cv.waitKey(0)
cv.destroyAllWindows()

'''
# Enhance contrast (helps makes boundaries clearer)
clache = cv.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
gray = clache.apply(gray)
ab = cv.resize(gray, (780, 617))
cv.imshow('image2',gray)
cv.waitKey(0)
cv.destroyAllWindows()
'''

## Identifies Cells
gaus = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,701,5)
ab = cv.resize(gaus, (780, 617))
cv.imshow('Binary Image',ab)
cv.waitKey(0)
cv.destroyAllWindows()

## Erodes border of the cells
ker_ero = np.ones((2,2),np.uint8)
eroded = cv.erode(gaus,ker_ero,iterations = 1)
ab = cv.resize(eroded, (780, 617))
cv.imshow('Eroded Image',ab)
cv.waitKey(0)
cv.destroyAllWindows()


'''
plt.subplot(1,3,1),plt.imshow(gray,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(gaus,cmap = 'gray')
plt.title('Gray'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(erosion,cmap = 'gray')
plt.title('Erosion'), plt.xticks([]), plt.yticks([])

plt.show()
'''
## Removes small white dots
#find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv.connectedComponentsWithStats(eroded, connectivity=4)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1]; nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 500

#your answer image
img2 = np.zeros(output.shape)
#for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255

ab = cv.resize(img2, (780, 617))
cv.imshow('White Dots Removed',ab)
cv.waitKey(0)
cv.destroyAllWindows()



# Finding sure foreground area
dist_transform = cv.distanceTransform(eroded,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.02*dist_transform.max(),255,0)
ab = cv.resize(sure_fg, (780, 617))
cv.imshow('image6',ab)
cv.waitKey(0)
cv.destroyAllWindows()

inv_img = np.uint8(255-img2)
ab = cv.resize(img2, (780, 617))
cv.imshow('Sure Fg',ab)
cv.waitKey(0)
cv.destroyAllWindows()
print(inv_img.dtype)

inv_sure_fg = np.uint8(255-sure_fg)
ab = cv.resize(inv_sure_fg, (780, 617))
cv.imshow('Sure Bg',ab)
cv.waitKey(0)
cv.destroyAllWindows()
print(inv_sure_fg.dtype)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(inv_sure_fg,inv_img)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]


ab = cv.resize(img, (780, 617))
cv.imshow('image6',ab)
cv.waitKey(0)
cv.destroyAllWindows()