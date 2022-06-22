import cv2
import numpy as np
import math


__author__ = "Alberto Abarzua"

def resize_transform(frame,des_width):
    """Resizes a frame to have widht = des_width mantaining it's ratio

    Args:
        frame (np.array): frame to resize
        des_width (int): desired width of new frame

    Returns:
        np.array: resized frame
    """
    scale = des_width/frame.shape[0]
    dims = (int(frame.shape[1]*scale),int(frame.shape[0]*scale))
    return cv2.resize(frame,dims,interpolation= cv2.INTER_AREA )

def crop(frame,des_width):
    """Crops a frame getting center square with des_widht sizeN

    Args:
        frame (np.array): frame
        des_width (int): with of output frame

    Returns:
        np.array: new cropped frame
    """
    width = frame.shape[1]
    cr_num = math.floor((width-des_width)/2)

    return frame[:, cr_num:-cr_num]


def prep_transform(frame,des_width):
    """Transform used to crop a frame and scale it down to a resolution of (des_width,des_with)

    Args:
        frame (np.array): frame to transform
        des_width (int): width,height of the new square frame

    Returns:
        np.array: new transformed frame.
    """
    return crop(resize_transform(frame,des_width),des_width)


def rescaleFrame(frame, scale=0.75):
    """Used to rescale a frame, mantaining it's ratio

    Args:
        frame (np.array): frame or image
        scale (float, optional): scale factor.. Defaults to 0.75.

    Returns:
        np.array: resized frame.
    """
    dimensions = (int(frame.shape[1] * scale),int(frame.shape[0] * scale))

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def frameTr(frame):
    """Frame transform used to get the diff between background and input frame

    Args:
        frame (np.array): frame to transform

    Returns:
        np.array: transformed frame.
    """
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
    if background is None:
        return frame.copy().astype("float")

    cv2.accumulateWeighted(frame,background,0.5)
    return background


def get_objs(background,frame,threshold,area_min):
    """Gets the biggest countours generated from the difference between the background frame and a given frame.

    Args:
        background (np.array): background frame reference
        frame (np.array): new frame where objects are
        threshold (int): threshhold value used to detect the counturs
        area_min (int): minimun area of contour to be considered

    Returns:
        list[tuple(int,int,int,int)],frame: tuple containig info of boxes that contain the resulting contours (x,y,w,h)
         and the difference frame between frame and background
    """
    frame = frameTr(frame)
    diff = cv2.absdiff(background.astype("uint8"),frame)
    _ ,thresh = cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)
    thresh = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
    contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) ==0:
        return None
    objs = sorted(contours,key = cv2.contourArea,reverse= True)
    objs = filter(lambda x: cv2.contourArea(x)>area_min,objs)
    return [cv2.boundingRect(obj) for obj in objs],(diff,thresh,contours,frame)



def get_roi(frame,box,size):
    """Gets the region of interest of frame that is containd in box, adding some extra margins with dimensions (size,size)

    Args:
        frame (np.array): frame to get the roi from
        box (tuple(int,int,int,int)): tuple of (x,y,w,h) to get the roi
        size (int): size of the resulting square frame (size,size)

    Returns:
        np.array: frame with roi
    """
    (x,y,w,h) = box
    r = max(w,h)
    cx,cy = (x+w//2,y+h//2)
    offset = r//2 +20
    y_min,y_max = max(0,cy-offset),min(size,cy + offset)
    x_min,x_max = max(0,cx-offset),min(size,cx + offset)
    frame =frame[y_min:y_max,x_min:x_max]
    return cv2.resize(frame,(32,32),interpolation= cv2.INTER_AREA )
