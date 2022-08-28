import cv2
import numpy as np

from .handlers import process


def main(input_image):
    
    #Read image
    image = cv2.imread(input_image)
    
    # Applaying Gaussian Blur on input image
    image_blurred = cv2.GaussianBlur(image, (25, 25),0)
    
    # Applaying Sobel Edge detections
    sobelX = cv2.Sobel(image_blurred ,cv2.CV_64F,1,0, ksize=-1)
    sobelY = cv2.Sobel(image_blurred,cv2.CV_64F,0,1, ksize=-1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    image_sobel = cv2.bitwise_or(sobelX, sobelY)
    

    # Change image from BGR to Gray
    imageGRAY = cv2.cvtColor(image_sobel, cv2.COLOR_BGR2GRAY)

    # Again applaying Gaussian Blur
    imageGRAY = cv2.GaussianBlur(imageGRAY, (7,7),0)

    # Change pixels greater than 0.3*max image pixel to 255
    imageGRAY[imageGRAY > 0.3*np.max(imageGRAY)] = 255

    # Change pixels values to 0 if is < 126 otherwise to 255
    _, imageGRAY = cv2.threshold(imageGRAY, 15, 255, cv2.THRESH_BINARY)
    # Merge pixels using opencv closing morphology functions
    image_morph = process.morph(imageGRAY, kernel=(3,3), show=False)
    # Remove smalls blobs
    image_conn = process.component(image_morph, min_size= 750, show=False)

    # Merge pixels using opencv closing morphology functions
    image_morph = process.morph(image_conn, kernel=(11,11), show=False)
    # Remove smalls blobs
    image_conn = process.component(image_morph, min_size= 750, show=False)

    # Normalize the pixel values in the image
    image_norm = cv2.normalize(image_conn, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # croping the image
    l, r = process.findroi(image_norm)
    image_norm_crop = image_norm[1:1024,l:r]

    # Merge pixels using opencv closing morphology functions
    image_conn_crop = process.component(image_norm_crop, min_size= 250, show=False)
    defect_area = process.find_contours(image_conn_crop)
    
    return defect_area