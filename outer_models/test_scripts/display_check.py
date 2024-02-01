import cv2

if __name__ == '__main__':
    
    image_path = '../img_test/1.png'

    img = cv2.imread(image_path)
    
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    
    cv2.imshow("window", img)
    cv2.waitKey(1)
    