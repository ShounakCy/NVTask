from preprocess.find_defect import main
import cv2 

if __name__ == '__main__':
    for i in range(1,4):
        input_image = './NVTask/'+str(i)+'.jpg'
        
        im = main(input_image)
        cv2.imwrite('./NVTask/defected_areas/defect_area'+str(i)+'.jpg', im)
 
