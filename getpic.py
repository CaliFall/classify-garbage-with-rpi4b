#-*- coding:utf-8 -*-
import cv2
 
cap=cv2.VideoCapture(0)
i=1
while(1):
    ret ,frame = cap.read()
    k=cv2.waitKey(1)
    if k==27:
        break
    elif k==32:
        cv2.imwrite('/home/pi/weightphotos_4/H/H_4_'+str(i)+'.jpg',frame)
        print("OK!"+"||"+str(i))
        i+=1
    cv2.imshow("capture", frame)
cap.release()
cv2.destroyAllWindows()
