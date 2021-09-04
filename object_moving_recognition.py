import cv2
import time
import imutils

cam = cv2.VideoCapture('Cars Moving On Road Stock Footage - Free Download.mp4')
time.sleep(1)

firstframe = None
area = 1000

while True:
    _,img = cam.read()
    text ="Normal"
    img=imutils.resize(img, width=900)
    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussianimg = cv2.GaussianBlur(grayimg, (41,41),0)
    if firstframe is None:
        firstframe = gaussianimg
        continue
    imgdiff = cv2.absdiff(firstframe, gaussianimg)
    threshimg = cv2.threshold(imgdiff,90,255,cv2.THRESH_BINARY)[1]
    threshimg = cv2.dilate(threshimg, None, iterations=4)
    cnts = cv2.findContours(threshimg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        text = "Moving object detected"
    print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,255),2)
    cv2.imshow('camerafeed',img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
