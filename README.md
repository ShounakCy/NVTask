# NVTask

## Directory Structure

```
NVTask/ ....  top src dir
|-- main.py ..................... main file to capture input and output image
|-- preprocess ............. image processing folder containing handlers to find defect 
|-- output_data ................ output images of defected data
|-- NN_method_extratest ................ Extra test done using CNN 
|-- requirement.txt ............. Requirements for building
```


## Requirements
```
Packages installation guide: ``pip install -r requirement.txt``

Please use Python3
```

## Usage
```
1. Run  `python main.py` to process the image data and the data will be stored into `output_data/`.
```

## How did you approach the problem?
```
### Steps Involved (check `process.jpg` flowchat)
`main.py`
1. Read input image
2. Applying Gaussian Blur on input image
3. Applying Sobel Edge detections
4. Converting image from BGR to Gray
5. Applying blurring again and thresholding the image
6. Merging pixels using opencv closing Morphology functions
7. Remove smalls blobs
8. Performing step 6 and 7 again with bigger kernel and increasing the minimum size of blob
9. Normalizing the pixel values in the image using MIN_MAX method, helps in changing the pixel intensity and increasing the overall contrast.
10. Cropping the required region ( this can be done at the veru starting as well, if we assume that the starting and ending pixel of the coating is same of all pixels)
11. Remove remaining small blobs
12. Output image with contours around the defect pathces and mask.

![alt text](https://github.com/ShounakCy/NVTask/blob/main/output_data/defect_area1.jpg)
![alt text](https://github.com/ShounakCy/NVTask/blob/main/output_data/defect_area2.jpg)
![alt text](https://github.com/ShounakCy/NVTask/blob/main/output_data/defect_area3.jpg)
```

## What are the main challenges you have faced?
```
1. Differentiating the foil and coating was difficult
2. Tuning was required to detect the defect in the third image
3. Getting the correct region of interest was difficult in order to draw contours around the patches itself not on the foil. So if the position is fixed it will be easier.
```

## What assumptions did you make?
```
 Given the number of images , while creating the ROI one assumption was if the sum of pixel of a column across all the rows are above a certain limit then its the boundary of the foil. Also calculated the moving sum of the pixels starting from the center both ways and calculated the maximum difference in a certain window. Whichever amongst the two has the lower pixel value is where the image gets cropped.
```

## What would you have done differently next time and with a larger dataset?
```
1. If proper dataset was provided, following steps could've been performed:
    - Preparing the segmentation masks of the picture data. Segmentation maps could be created. The segmentation map should be able to identify the image's problematic areas.Therefore, to mark and annotate these regions as defects, we can make use of ellipses.
    - Then UNet model, could be used for image segmentation tasks. The model will be able to localize the acquired context using upsampling techniques and comprehend the shapes, patterns, edges, and other features contained in the image. The encoder which is a stack of CNN and pooling layers helps in understanding the and capture the contexts in the image, while the decoder is used to localise the the captured contexts.

2. Tested with a simple CNN model by genearting new images from the defected images, to detect if the correct image is defective or not. `NN_method_extratest`
```

## What external factors would you change to make the task easier (camera, lighting, etc.)
```
In order to perform opencv techniques, I would keep the camera position fixed so that the croping of the coating could be fixed and constant illumination from the sides to avoid refecltion. I would keep coaxial forward lighting for detecting the defects, as it provides more uniform illumination than traditional lighting mode, while avoiding the reflection of the object.
```