from preprocess.find_defect import main
import cv2
if __name__ == '__main__':
    for i in range(1,2):
        input_image = './NVTask/'+str(i)+'.jpg'
        
        defect = main(input_image)
        cv2.imwrite('./NVTask/defected_data/defect_area'+str(4)+'.jpg', defect)
        
 
