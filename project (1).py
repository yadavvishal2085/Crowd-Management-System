import cv2
import torch
import numpy as np
pointd=[]
def get_coordinates(event, x, y, flags, param):
    if(event==cv2.EVENT_MOUSEMOVE):
        cordinates=[x,y]
        print(cordinates)
cv2.namedWindow("FRAME")
cv2.setMouseCallback('FRAME',get_coordinates)
model=torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)
cap=cv2.VideoCapture('people.mp4')
count=0
area=[(306,42),(263,428),(967,428),(785,42)]
while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(1020,600))
    result=model(frame)
    list=[]
    for index,row in result.pandas().xyxy[0].iterrows():
        x1=int(row['xmin'])
        y1=int(row['ymin'])
        x2=int(row['xmax'])
        y2=int(row['ymax'])
        d=(row['name'])
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        if 'person' in d:
            results=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
            if results>=0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                list.append([cx])
    cv2.polylines(frame,[np.array(area,np.int32)],True,(0,255,0),2)
    a=len(list)
    cv2.putText(frame,"Count "+str(a),(50,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    count=0
    for itr in list:
            
        count=count+1
        

    if(count>16):
        cv2.putText(frame,"Over Crowded",(50,80),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2)
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()