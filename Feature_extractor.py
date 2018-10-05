
import numpy as np
import cv2
from scipy.stats import skew, kurtosis, entropy
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern

## Converts the image to grayscale and create a histogram
def GrayScale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [128], [0, 256])
    '''
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()
    '''
    return np.squeeze((np.array(cv2.normalize(hist, hist))))

## Colour Histogram
def ColourHist(img,bins):
    col = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    chans = np.array(cv2.split(col))

    # Normalize image
    sum = chans[0]+chans[1]+chans[2]
    #print(sum)
    chans[0] = np.divide(chans[0],sum) * 255
    chans[1] = np.divide(chans[1],sum) * 255
    chans[2] = np.divide(chans[2],sum) * 255
    #del chans[0]

    #print(chans[0])
    #colors = ('g','r')
    #print('Length list: ',len(chans),len(chans[0]))
    '''
    plt.figure()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    '''

    hist = cv2.calcHist([chans[1],chans[2]], [0,1], None, [bins,bins], [0, 256, 0, 256])
    hist = cv2.normalize(hist, hist)

 #       plt.plot(hist, color=color)
  #      plt.xlim([0, bins])

    #print("flattened feature vector size: %d" % (np.array(features).flatten().shape))
  #  plt.show()

    return np.array(hist.flatten())


## Colour Histogram
def ColourHist2(img, bins,channel):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize image
    sum = img[0] + img[1] + img[2]
    # print(sum)
    img[0] = np.divide(img[0], sum) * 255
    img[1] = np.divide(img[1], sum) * 255
    img[2] = np.divide(img[2], sum) * 255

    features = []
    hist = cv2.calcHist([img[channel]], [0], None, [bins], [0, 256])
    features.extend(hist)

    hist = cv2.normalize(hist, hist)
    return np.array(hist.flatten())


## Saturation Level Histogram
def SaturationHist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    '''
    plt.figure()
    plt.title("Saturation Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()
    '''
    return np.squeeze(np.array(cv2.normalize(hist, hist)))


def SobelHistX(img):
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    hist = cv2.calcHist([sobelx], [0], None, [256], [0, 256])
    hist = np.squeeze(cv2.normalize(hist, hist))
    #print(hist_x.shape)

    return hist

def SobelHistY(img):
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    hist = cv2.calcHist([sobely], [0], None, [256], [0, 256])
    hist = np.squeeze(cv2.normalize(hist, hist))
    #print(hist_y.shape)

    return hist

def LaplacianHist(img):
    # Calcution of Sobelx
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_32F)

    hist = cv2.calcHist([lap], [0], None, [64], [0, 256])

    return np.squeeze((np.array(cv2.normalize(hist, hist))))

def YCBHist(img,bins):
    YCB = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hist = cv2.calcHist([YCB[1],YCB[2]], [0,1], None, [bins,bins], [0, 256,0, 256])
    hist = cv2.normalize(hist, hist)

    return np.array(hist.flatten())

def LBPHist(img,radius,points):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    no_points = points * radius

    # Uniform LBP is used
    lbp = local_binary_pattern(gray, no_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    (hist, _) = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
    eps = 1e-7

    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    return hist

def HuMoment(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.HuMoments(cv2.moments(img)).flatten()

    return img

def ContrastHist(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])         # jumped from 84.9% to 88.46% with 128 bins. with 64 achieved 87%, 256 is the best

    return np.squeeze((np.array(cv2.normalize(hist, hist))))

def Contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    #var = np.var(gray)
    #M4 = kurtosis(gray)

    return entropy(hist) #var/pow(M4,0.25)

def FeatureReduction(numpy):

    features = []

    features.append(np.mean(numpy))
    features.append(np.std(numpy))
    features.append(skew(numpy))
    features.append(kurtosis(numpy))
    features.append(entropy(numpy))

    features = np.array(features)

    return features

def ObtainFeatures(X,num):

    ## Obtains Colour Histogram Red Channel
    a = []
    for i in range(0, num):
        a.append(ColourHist2(X[i],256,0))  # 24 is the best for Colo

    a = np.array(a)

    # Feature reduction Colour Histogram Red Channel
    b = []
    for i in range(0,num):
        b.append(FeatureReduction(a[i]))


    features = np.array(b)
    print('Color Red: ', len(features[0]))

    ## Obtains Colour Histogram Green Channel
    a = []
    for i in range(0, num):
        a.append(ColourHist2(X[i],256,1))  # 24 is the best for Colo

    a = np.array(a)

    # Feature reduction Colour Histogram Green Channel
    b = []
    for i in range(0,num):
        b.append(FeatureReduction(a[i]))

    b = np.array(b)
    print('Color Green: ', len(b[0]))
    features = np.concatenate((features, b), axis=1)

    ## Obtains Saturation Histogram
    a = []
    for i in range(0, num):
        a.append(SaturationHist(X[i]))

    a = np.array(a)

    # Feature reduction Saturation Histogram
    b = []
    for i in range(0,num):
        b.append(FeatureReduction(a[i]))

    b = np.array(b)
    print('Saturation: ', len(b[0]))
    features = np.concatenate((features, b), axis=1)

        ## Obtains GrayScale Histogram
    a = []
    for i in range(0, num):
        a.append(GrayScale(X[i]))

    a = np.array(a)

    # Feature reduction GrayScale Histogram
    b = []
    for i in range(0,num):
        b.append(FeatureReduction(a[i]))

    b = np.array(b)
    print('GrayScale: ', len(b[0]))
    features = np.concatenate((features, b), axis=1)


    ## Obtains Sobel Horizontal(X) Histogram
    a = []
    for i in range(0, num):
        a.append(SobelHistX(X[i]))

    a = np.array(a)

    # Feature reduction Sobel Horizontal Histogram
    b = []
    for i in range(0,num):
        b.append(FeatureReduction(a[i]))

    b = np.array(b)
    print('Sobel X: ', len(b[0]))
    features = np.concatenate((features, b), axis=1)

    ## Obtains Sobel Vetical(Y) Histogram
    a = []
    for i in range(0, num):
        a.append(SobelHistY(X[i]))

    a = np.array(a)

    # Feature reduction Sobel Vetical Histogram
    b = []
    for i in range(0,num):
        b.append(FeatureReduction(a[i]))

    b = np.array(b)
    print('Sobel Y: ', len(b[0]))
    features = np.concatenate((features, b), axis=1)


    ## Enhancing contrast Image Histogram
    a = []
    for i in range(0, num):
        a.append(ContrastHist(X[i]))

    a = np.array(a)

    #Feature reduction Enhancing contrast Image Histogram
    b = []
    for i in range(0,num):
        b.append(FeatureReduction(a[i]))

    b = np.array(b)
    print('Enhancing Contrast: ', len(b[0]))
    features = np.concatenate((features,b), axis=1) #np.concatenate((features, b), axis=1)

    '''
    ## Obtains LBP Histogram
    a = []
    for i in range(0, num):
        a.append(LBPHist(X[i], 2, 8))  # 2,8 Acc ->84.7%

    a = np.array(a)
    print('LBP X: ', len(a[0]))
    features = np.concatenate((features, a), axis=1)
    '''

    '''
    ## Obtains Laplacian Histogram
    a = []
    for i in range(0, num):
        a.append(LaplacianHist(X[i]))

    a = np.array(a)

    # Feature reduction Laplacian Histogram
    b = []
    for i in range(0,num):
        b.append(FeatureReduction(a[i]))

    b = np.array(b)

    features = np.concatenate((features, b), axis=1)
    '''

    '''
    ## Obtains YCrBr Histogram
    a = []
    for i in range(0, num):
        a.append(YCBHist(X[i],2))

    a = np.array(a)

    # Feature reduction Saturation Histogram
    b = []
    for i in range(0,num):
        b.append(FeatureReduction(a[i]))

    b = np.array(b)
    print('YCrBr: ', len(b))
    features = np.concatenate((features, b), axis=1)
    '''

    '''
    ## Obtains Colour Histogram
    a = []
    for i in range(0, num):
        a.append(ColourHist(X[i],24))  # 24 is the best for Colo

    a = np.array(a)

    # Feature reduction Colour Histogram
    b = []
    for i in range(0,num):
        b.append(FeatureReduction(a[i]))

    b = np.array(b)
    print('Color: ', len(b[0]))
    features = np.concatenate((features, b), axis=1)
    '''

    '''
    ## Obtains Colour Histogram Blue Channel
    a = []
    for i in range(0, num):
        a.append(ColourHist2(X[i],256,2))  # 24 is the best for Colo

    a = np.array(a)

    # Feature reduction Colour Histogram Blue Channel
    b = []
    for i in range(0,num):
        b.append(FeatureReduction(a[i]))

    b = np.array(b)
    print('Color Red: ', len(b[0]))
    features = np.concatenate((features, b), axis=1)
    '''

    '''
      ## Obtains Contrast Histogram

    a = []
    for i in range(0, num):
        a.append(Contrast(X[i]))

    a = np.array(a)
    print('Enhancing Contrast: ', len(a))

    features = np.concatenate((features, a), axis=1)
    '''
    return features
