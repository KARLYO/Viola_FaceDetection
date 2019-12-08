import cv2 as cv
picname = "D:/dataset/serie.jpg"

def facedetect(picname):
     

    face_cascade = cv.CascadeClassifier('D:/dataset/haarcascade_frontalface_default.xml')
    
 
    img = cv.imread(picname)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     
    BBox = face_cascade.detectMultiScale(gray, 1.3, 5)
 
    for(x, y, w, h) in BBox:
        img = cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
   
    cv.namedWindow('FaceDetection')
    cv.imshow('FaceDetection', img)
    
    cv.imwrite('D:/dataset/result/serie.jpg', img)
    cv.waitKey(0)

facedetect(picname)
