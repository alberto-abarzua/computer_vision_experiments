import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def resize_transform(frame,des_width):

    scale = des_width/frame.shape[0]
    dims = (int(frame.shape[1]*scale),int(frame.shape[0]*scale))
    return cv2.resize(frame,dims,interpolation= cv2.INTER_AREA )

def crop(frame,des_width):
    width = frame.shape[1]
    cr_num = math.floor((width-des_width)/2)

    return frame[:, cr_num:-cr_num]


def prep_transform(frame,des_width):
    return crop(resize_transform(frame,des_width),des_width)


def rescaleFrame(frame, scale=0.75):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def frameTr(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    kernel = np.ones((4,4),np.float32)/27
    frame = cv2.filter2D(frame,-1,kernel)
    frame = cv2.normalize(frame,  frame, 0, 255, cv2.NORM_MINMAX)
    return frame

def define_background(background,frame):
    """Used to create the initial background frame to later detect new objects

    Args:
        background (np.array): img that stores the background
        frame (np.array): new frame we want to add to the background.

    Returns:
        np.array: modified background, or frame if background was None.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if background is None:
        return frame.copy().astype("float")

    cv2.accumulateWeighted(frame,background,0.5)
    return background


def get_objs(background,frame,threshold,area_min):
    frame = frameTr(frame)
    diff = cv2.absdiff(background.astype("uint8"),frame)
    _ ,thresh = cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) ==0:
        return None
    objs = sorted(contours,key = cv2.contourArea,reverse= True)
    objs = filter(lambda x: cv2.contourArea(x)>area_min,objs)
    return [cv2.boundingRect(obj) for obj in objs],diff

cam = cv2.VideoCapture(0)

def get_roi(frame,box,size):
    (x,y,w,h) = box
    r = max(w,h)
    cx,cy = (x+w//2,y+h//2)
    offset = r//2 +20
    y_min,y_max = max(0,cy-offset),min(size,cy + offset)
    x_min,x_max = max(0,cx-offset),min(size,cx + offset)
    frame =frame[y_min:y_max,x_min:x_max]
    return cv2.resize(frame,(32,32),interpolation= cv2.INTER_AREA )


num_frames = 0
background = None
FRAME_SIZE = 128
while True:
    ret, pre_frame = cam.read()

    frame = prep_transform(pre_frame,des_width = FRAME_SIZE)
    frame_copy = frame.copy()

   
    if num_frames < 60:
        background = define_background(background, frame_copy)
        if num_frames <= 59:
            cv2.putText(frame, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("obj",frame)
        num_frames += 1
        continue
            
    
    val = get_objs(background,frame_copy,65,200)
    if val is not None: 
        objs,_= val
        for elem in objs:
            (x,y,w,h) = elem
            cv2.imshow("obj1", rescaleFrame(get_roi(frame,elem,FRAME_SIZE),5))
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)

    num_frames += 1

    cv2.imshow("obj", rescaleFrame(frame,5))


    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()