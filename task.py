import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rng

rng.seed(12345)

def morph(image, kernel = (5, 5), show=False):
    element_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element_closing)
    if show:
        cv2.imshow(f'Kernel: {kernel}', img)
    return img

def component(image, connectivity = 8, min_size = 100, show=False):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity= connectivity)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    img = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img[output == i + 1] = 255
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    if show:
        cv2.imshow(f'Min size: {min_size}', img)
    return img

def findroi(img):
    mid_pixel = int(len(img[1])/2)
    l_x=[]
    l_y=[]
    r_x=[]
    r_y=[]
    
    for i in range(1, mid_pixel):
        
        left_pixel = mid_pixel -i
        sumleftleft = int(np.sum(img[:,left_pixel-4:left_pixel-2]))
        sumleft = int(np.sum(img[:,left_pixel-2:left_pixel]))
        diff = abs(sumleft-sumleftleft)
        l_x.append(i)
        l_y.append(diff)    
        
        if sumleftleft > 0.66*255*2*1024 :
            left = left_pixel+1
            break
        else:
            l_pixel = l_y.index(np.max(l_y))
            left = 1 + (mid_pixel - l_pixel)
                
        i+=1
   
    for j in range(1, mid_pixel):
        
        right_pixel = mid_pixel +j
        sumrightright = int(np.sum(img[:,right_pixel+2:right_pixel+4]))
        sumright = int(np.sum(img[:,right_pixel:right_pixel+2]))
        diff = abs(sumright-sumrightright)
        r_x.append(j)
        r_y.append(diff)
        
        if sumrightright > 0.66*255*2*1024 :
            right = right_pixel-1
            break
        else:
            right = mid_pixel + r_y.index(np.max(r_y)) -1
        
        j+=1
    
    return left, right

def find_contours(image_conn_crop):

    contours, hierarchy = cv2.findContours(image_conn_crop.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("Number of Contours found = " + str(len(contours)))
    #mask = np.zeros(image_conn_crop.shape, np.uint8)
    boundRect = [None]*len(contours)
    contours_poly = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    mask = np.zeros((image_conn_crop.shape[0], image_conn_crop.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv2.drawContours(mask, contours_poly, i, color, thickness=cv2.FILLED)
            #cv2.rectangle(mask, (int(boundRect[i][0]), int(boundRect[i][1])), \
            #  (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            cv2.circle(mask, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
            
    return mask

def find_defect(image):
    # Applaying Gaussian Blur on input image
    image_blurred = cv2.GaussianBlur(image, (25, 25),0)
    #cv2.imwrite("image_blurred.png", image_blurred)
    # Applaying Sobel Edge detections
    sobelX = cv2.Sobel(image_blurred ,cv2.CV_64F,1,0, ksize=-1)
    sobelY = cv2.Sobel(image_blurred,cv2.CV_64F,0,1, ksize=-1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    image_sobel = cv2.bitwise_or(sobelX, sobelY)
    #cv2.imwrite("image_sobel.png", image_sobel)

    # Change image from BGR to Gray
    imageGRAY = cv2.cvtColor(image_sobel, cv2.COLOR_BGR2GRAY)

    #cv2.imwrite("imageGRAY.png", imageGRAY)
    # Again applaying Gaussian Blur
    imageGRAY = cv2.GaussianBlur(imageGRAY, (7,7),0)
    #cv2.imwrite("imageGRAY.png", imageGRAY)

    # Change pixels greater than 0.3*max image pixel to 255
    imageGRAY[imageGRAY > 0.3*np.max(imageGRAY)] = 255

    # Change pixels values to 0 if is < 126 otherwise to 255
    _, imageGRAY = cv2.threshold(imageGRAY, 15, 255, cv2.THRESH_BINARY)
    #cv2.imwrite("imageGRAY.png", imageGRAY)
    # Merge pixels using opencv closing morphology functions
    image_morph = morph(imageGRAY, kernel=(3,3), show=False)
    #cv2.imwrite("age_morph.png", image_morph)
    # Remove smalls blobs
    image_conn = component(image_morph, min_size= 750, show=False)

    # Merge pixels using opencv closing morphology functions
    image_morph = morph(image_conn, kernel=(11,11), show=False)
    # Remove smalls blobs
    image_conn = component(image_morph, min_size= 750, show=False)
    #cv2.imwrite("image_conn.png", image_conn)
    image_norm = cv2.normalize(image_conn, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    #cv2.imwrite("image_norm.png", image_norm)

    l, r = findroi(image_norm)
    image_norm_crop = image_norm[1:1024,l:r]
    #cv2.imwrite("/efs/workspace/nvtest/image_norm_crop.png", image_norm_crop)
    # Merge pixels using opencv closing morphology functions
    #image_morph = morph(image_norm_crop, kernel=(11,11), show=False)
    image_conn_crop = component(image_norm_crop, min_size= 250, show=False)
    #cv2.imwrite("/efs/workspace/nvtest/image_conn_crop.jpg", image_conn_crop)
    defect_area = find_contours(image_conn_crop)
    
    return defect_area
 
if __name__ == '__main__':
    for i in range(1,4):
        image = cv2.imread('/efs/workspace/NVTask/'+str(i)+'.jpg')
        defect_area = find_defect(image)
        cv2.imwrite('/efs/workspace/NVTask/defect_area'+str(i)+'.jpg', defect_area)
