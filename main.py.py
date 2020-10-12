import cv2
import math
import numpy as np
import skimage.morphology
import matplotlib.pyplot as plt
import joblib
import numpy as np
import math
import tensorflow
from tensorflow import keras
import glob
import sys
from skimage.morphology import square

args = sys.argv[1:]

#We load our Convolutionnal Neural Network Classifier
filename = 'digits_cnn2-Copy2.joblib.pkl'
clf = joblib.load(filename)

#first we get all the frames from the video
img_array = []
cap = cv2.VideoCapture(args[0])
if (cap.isOpened()== False):
  print("Error opening video stream or file")
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img_array.append(frame)
    if ret != True:
        break

cap.release()
#We create a list for the frames of the output video
img_out=[]

#We take the first frame to analyze it
imtoan=img_array[0]

# we find the center of the red arrow by isolating it by color
hsv=cv2.cvtColor(imtoan,cv2.COLOR_BGR2HSV)
lower_range=np.array([0,100,40])
upper_range=np.array([300,300,350])

mask = cv2.inRange(hsv, lower_range, upper_range)

# we do opening to remove small feature appearing in red
opening=skimage.morphology.area_opening(mask,area_threshold=200, connectivity=1);

lower_range2=np.array([0,0,0])
upper_range2=np.array([200,300,140])
 
mask1 = cv2.inRange(hsv, lower_range2, upper_range2)
mask2 = skimage.morphology.area_opening(mask1,area_threshold=300, connectivity=1) #open such as the operators disappear
mask2 = skimage.morphology.dilation(mask2,square(30)); # dilate the rest and substract to obtain a cleanb image

mask1=mask1-mask2

# Find contours in the image in the opening to localize the arrow
ctrs, hier = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the rectangle contouring the arrow to get its center
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
p=math.floor(rects[0][2]/2)
s=math.floor(rects[0][3]/2)
fleche=[rects[0][0]+p,rects[0][1]+s]
pointe=np.array([[rects[0][0]+p,rects[0][1]+s]])

im = imtoan.copy()

# Convert to grayscale and apply Gaussian filtering the first frame
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
afleche = (fleche[0], fleche[1])

# Threshold the image
#ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
im_th=mask1
# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# We isolate the contour of each operator and merge the contours next to each other
for ind1, j in enumerate(rects):
    for ind2, i in enumerate(rects):
        if (i[0] != j[0] and i[1] != j[1] and j[2] > 1 and j[3] > 1 and i[3] > 1 and i[2] > 1):
            dist = math.sqrt(((j[0] - i[0]) ** 2) + ((j[1] - i[1]) ** 2))
            distf = math.sqrt(((fleche[0] - i[0]) ** 2) + ((fleche[1] - i[1]) ** 2))
            if (distf < 70):
                rects.remove(i)
            iteml1 = list(i)
            iteml2 = list(j)
            if (dist < 15):
                # j[0]=(j[0]+i[0])/2
                # j[1]=(j[1]+i[1])/2
                # j[2]=j[2]+abs(j[0]-i[0])
                # j[3]=j[3]+abs(j[1]-i[1])
                if (j[0] < i[0]):
                    iteml1[0] = j[0]
                else:
                    iteml1[0] = i[0]

                if (j[1] < i[1]):
                    iteml1[1] = j[1]
                else:
                    iteml1[1] = i[1]
                iteml1[2] = 20
                # int((j[2]+i[2])/2)
                iteml1[3] = 20
                # int((j[3]+i[3])/2)
                item = tuple(iteml1)
                rects[ind1] = item
                rects.remove(i)

# We remove some invariance we have
rects = [rect for rect in rects if
         (rect[2] < 40 and rect[3] < 40 and rect[2] > 8 and rect[3] > 8 and rect[1] < 400 and rect[0] > 100)]

# From these rectangles we create images 28x28 to analyze
lin = []
for rect in rects:
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, np.ones((2, 2), np.uint8))
    lin.append(roi.tolist())

vu=[]
#We perform our classifier on the images to get their labels
#If we dont sufficient results, we rotate the images
for i in lin:
    best=0
    num=0
    k=0
    while(best<0.8):
        rows,cols=np.array(i).shape
        M=cv2.getRotationMatrix2D((rows/2,cols/2),k,1)
        f=np.array(i,np.uint8)
        v=cv2.warpAffine(f,M,(cols,rows))
        j=np.array([v.tolist()])
        j = j.reshape((j.shape[0], 28, 28, 1)).astype('float32')
        j=j/255
        res=clf.predict(j)
        for c,value in enumerate(res[0],0):
            if(value>best):
                best=value
                num=c
            if(c==11 and value>0.1):
                best=1
                num=c
        k=k+2
    vu.append(num)

# We print the number to show that we identfied them
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2
results = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '6', '+', '-', '/', '*', '=']
for c, rect in enumerate(rects, 0):
    # Using cv2.putText() method
    image = cv2.putText(im, results[vu[c]], (rect[0], rect[1]), font,
                        fontScale, color, thickness, cv2.LINE_AA)

pr = 15
showr = []
showp = []
eq = ''
showp.append((fleche[0], fleche[1]))
# We now take all the remaining frames and do the same to identify the arrow
# If the the arrow is next to a number, he crosses it and we register that
# We write at the same time the equation at the bottom
for i in range(1, len(img_array) - 1):
    hsv = cv2.cvtColor(img_array[i], cv2.COLOR_BGR2HSV)
    lower_range = np.array([0, 100, 40])
    upper_range = np.array([300, 300, 350])

    mask = cv2.inRange(hsv, lower_range, upper_range)
    res = cv2.bitwise_and(imtoan, imtoan, mask=mask)
    opening = skimage.morphology.area_opening(mask, area_threshold=200, connectivity=1);
    # Find contours in the image
    ctrs, hier = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rec = [cv2.boundingRect(ctr) for ctr in ctrs]
    p = math.floor(rec[0][2] / 2)
    s = math.floor(rec[0][3] / 2)
    fleche = [rec[0][0] + p, rec[0][1] + s]
    showp.append((fleche[0], fleche[1]))
    for a in range(len(showp) - 1):
        image = cv2.line(img_array[i], showp[a], showp[a + 1], (0, 0, 255), thickness, cv2.LINE_AA)

    for rect in rects:
        image = cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

    for c, rect in enumerate(rects, 0):
        # Using cv2.putText() method
        image = cv2.putText(image, results[vu[c]], (rect[0], rect[1]), font,
                            fontScale, color, thickness, cv2.LINE_AA)

    for c, rect in enumerate(rects):
        distf = math.sqrt(((fleche[0] - rect[0]) ** 2) + ((fleche[1] - rect[1]) ** 2))
        if (distf < 30 and pr != c):
            showr.append(results[vu[c]])
            if (vu[c] != 14):
                eq = eq + results[vu[c]]
            pr = c

    k = 50
    for b in showr:
        image = cv2.putText(image, b, (k, image.shape[0] - 30), font,
                            fontScale, color, thickness, cv2.LINE_AA)
        k += 30
    if (i == len(img_array) - 2):
        image = cv2.putText(image, str(eval(eq)), (k, image.shape[0] - 30), font,
                            fontScale, color, thickness, cv2.LINE_AA)
        k += 30

    img_out.append(image.copy())

height, width, layers = img_array[0].shape
size = (width, height)
out = cv2.VideoWriter(args[1], cv2.VideoWriter_fourcc(*'DIVX'), 2, size)

for i in range(len(img_out)):
    out.write(img_out[i])
out.release()