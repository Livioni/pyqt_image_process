from time import sleep
import cv2  
import numpy as np  
  
  
xs,ys,ws,hs = 0,0,0,0  #selection.x selection.y  
xo,yo=0,0 #origin.x origin.y  
selectObject = False  
trackObject = 0  
def onMouse(event, x, y, flags, prams):   
    global xs,ys,ws,hs,selectObject,xo,yo,trackObject  
    if selectObject == True:  
        xs = min(x, xo)  
        ys = min(y, yo)  
        ws = abs(x-xo)  
        hs = abs(y-yo)  
    if event == cv2.EVENT_LBUTTONDOWN:  
        xo,yo = x, y  
        xs,ys,ws,hs= x, y, 0, 0  
        selectObject = True  
    elif event == cv2.EVENT_LBUTTONUP:  
        selectObject = False  
        trackObject = -1  
  
cap = cv2.VideoCapture('videos/pets2001.avi')  
ret,frame = cap.read()  
cv2.namedWindow('imshow')  
cv2.setMouseCallback('imshow',onMouse)  
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )  
while(True):  
    ret,frame = cap.read()  
    if frame is not None:
        if trackObject != 0:  
            hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
            mask = cv2.inRange(hsv, np.array((0., 30.,10.)), np.array((180.,256.,255.)))  
            if trackObject == -1:  
                track_window=(xs,ys,ws,hs)  
                maskroi = mask[ys:ys+hs, xs:xs+ws]  
                hsv_roi = hsv[ys:ys+hs, xs:xs+ws]  
                roi_hist = cv2.calcHist([hsv_roi],[0],maskroi,[180],[0,180])  
                cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)  
                trackObject = 1  
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)  
            dst &= mask  
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)  
            pts = cv2.boxPoints(ret)  
            pts = np.int0(pts)  
            img2 = cv2.polylines(frame,[pts],True, 255,2)  
            
        if selectObject == True and ws>0 and hs>0:  
            cv2.imshow('imshow1',frame[ys:ys+hs,xs:xs+ws])  
            cv2.bitwise_not(frame[ys:ys+hs,xs:xs+ws],frame[ys:ys+hs,xs:xs+ws])  
        cv2.imshow('imshow',frame)  
        if  cv2.waitKey(10)==27:  
            break  
    else:
        break
cv2.destroyAllWindows()  