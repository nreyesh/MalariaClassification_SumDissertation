##----------------------------------------------------------------##
##------------ Takes care of clean Parasitized Images ------------##

'''
import numpy as np
import cv2 as cv
import os

## Loading Paths
dir_img = '/home/nicor/Documents/Summer_Project/malaria-full/Train/Parasitized'
img_names = os.listdir(dir_img)
img_names.sort()
print(img_names)

i=0
save = '/home/nicor/Documents/Summer_Project/Final_Test_Images/Parasitized/'
for names in img_names:
    #print(os.path.join(dir_img,img_names[i]))
    img = cv.imread(os.path.join(dir_img,names), cv.IMREAD_COLOR)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #cv.imshow('White Dots Removed',img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()


    ## Identifies Cells
    gaus = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,701,5)


    ## Removes small white dots
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(gaus, connectivity=4)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size =400

    #your answer image
    img2 = np.zeros(output.shape)
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    # Inverts Image
    inv_img = np.uint8(255-img2)

    ## Removes small white dots
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(inv_img, connectivity=4)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    #min_size =400

    #your answer image
    img2 = np.zeros(output.shape)
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    #cv.imshow('White Dots Removed',img2)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    ## Puting Black Backgroung
    img2 = np.uint8(img2)
    img2 = np.stack((img2,)*3, -1)
    #print(img2.shape)

    # Convert uint8 to float
    foreground = img.astype(float)
    background = img2.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = img2.astype(float) / 255
    #print(foreground.shape)
    #print(background.shape)
    #print(alpha.shape)

    # Multiply the foreground with the alpha matte
    foreground = cv.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    outImage = cv.add(foreground, background)

    cv.imwrite(save + names, outImage)

    i+=1

    # Display image
    #cv.imshow("outImg", outImage / 255)
    #cv.waitKey(100)
    #cv.destroyAllWindows()

'''


##----------------------------------------------------------------##
##------------ Takes care of clean Uninfected Images ------------##


import cv2
import numpy as np
import os

## Loading Paths
dir_img = '/home/nicor/Documents/Summer_Project/malaria-full/Train/Uninfected'
img_names = os.listdir(dir_img)
img_names.sort()
#print(img_names)

save = '/home/nicor/Documents/Summer_Project/Final_Test_Images/Uninfected/'
for names in img_names:
    im = cv2.imread(os.path.join(dir_img, names), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #im = cv2.imread('/home/nicor/Documents/Summer_Project/malaria-full/Train/Uninfected/img4_11.jpg')
    #gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    _, bin = cv2.threshold(gray,120,255,1) # inverted threshold (light obj on dark bg)
    bin = cv2.dilate(bin, None)  # fill some holes
    bin = cv2.dilate(bin, None)
    bin = cv2.erode(bin, None)   # dilate made our shape larger, revert that
    bin = cv2.erode(bin, None)
    bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rc = cv2.minAreaRect(contours[0])
    box = np.array(cv2.boxPoints(rc),dtype='int_')
    #print(box.shape)
    #print(box,'\n')
    #cv2.imshow("plank", im)
    #cv2.waitKey()

    xi = np.amin(box[:,0])
    xf = np.amax(box[:,0])
    yi = np.amin(box[:,1])
    yf = np.amax(box[:,1])

    #print(xi,xf)
    #print(yi,yf)

    cell = gray[yi:yf,xi:xf]

    cv2.imwrite(save + names+'.jpg', cell)
    #cv2.imshow("outImg", cell)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()