import cv2
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
vid= cv2.VideoCapture(0)
fd = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
sd= cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')
notCaptured = True
while notCaptured:
    flag , img = vid.read()
    if flag:
        # processing code
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # print(type(img_gray))
        # break
        faces = fd.detectMultiScale(img_gray,
                                    scaleFactor=1.1,
                                    minNeighbors=5,
                                    minSize=(50,50))
        
        
        np.random.seed(50)
        colors = np.random.randint(0,255,(len(faces),3)).tolist() 
        i=0
        
        for x,y,w,h in faces:
            face = img_gray[y:y+h,x:x+w].copy()
            smiles=sd.detectMultiScale(face,scaleFactor=1.1,
                                    minNeighbors=5,)
            print(len(smiles))
            
            if len(smiles)==1:
                cv2.imwrite('myselfie.png',img)
                notCaptured = False
                break


            cv2.rectangle(img,pt1=(x,y), pt2=(x+w,y+h), color=colors[i]
                          ,thickness=5)   
            i+=1

       
                
       
        cv2.imshow('Preview',img)

        # cv2.imshow('Preview',img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
            
    else: 
        print('N o Frames')
        break
    sleep(0.1)
vid.release()      
cv2.destroyAllWindows()
 