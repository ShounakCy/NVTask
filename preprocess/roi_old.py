from operator import and_
import sys
from PIL import Image
import numpy as np
from numpy import asarray
import cv2

#open image
image = Image.open('.NVTask/ok/1.jpg');

#rigid body dimensions cal
img_width = int(image.size[0])
img_height = int(image.size[1])
img_width_mid = int(img_width/2)
img_height_mid = int(img_height/2)

#convert into integer np array
imgarr = asarray(image)

#traversal array holding column add values
height_avg = np.zeros(img_width)

#variables holding edge values
right_edge = 0
left_edge = 0

#logic
for i in range((img_width_mid-1),0,-1):
    for j in range(0,img_height):
        height_avg[i] += imgarr[j][i]
    #print(i,height_avg[i])
    if( ( i < (img_width_mid-1) ) and ( (height_avg[i] - height_avg[i+1]) < -6000 ) ):
        left_edge = i
        break

for i in range(img_width_mid,img_width):
    for j in range(0,img_height):
        height_avg[i] += imgarr[j][i]
    #print(i,height_avg[i])
    if( (i > img_width_mid) and ( (height_avg[i] - height_avg[i-1]) < -6000 ) ):
        right_edge = i
        break

#np array to extract coating 
newimg = np.zeros(shape=(img_height,(right_edge - left_edge) ), dtype=np.uint8)
for i in range(left_edge,right_edge):
    for j in range(0,img_height):
        newimg[j][i-left_edge] = imgarr[j][i]

#converting the np array in image
data = Image.fromarray(newimg)
im1 = data.save("/efs/workspace/nvtest/data.jpg")
#data.show()

#add code below for saving the coating image
