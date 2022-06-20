
from configparser import Interpolation
import cv2 as cv
import shutil
import os
from pathlib import Path
from transforms import *

#Paths  
BASE = Path(__file__).parent
DATA = BASE / "data"
if os.path.exists(DATA): #Clear the directory
    shutil.rmtree(DATA)

os.mkdir(DATA) #Create fresh dir.


# img = cv.imread(DATA.joinpath("frame{}.jpg".format(1)).__str__())


# cv.imshow(img, img)

vid = cv.VideoCapture(0)
count =0
while(True):
    succes, pre_frame = vid.read()
    frame = prep_transform(pre_frame,des_width = 128)
    # Display the resulting frame
    cv.imshow('frame', frame)
    print(f"Frame size is ({frame.shape[1]} , {frame.shape[0]})")
    cv.imwrite(DATA.joinpath("frame{}.jpg".format(count)).__str__(), frame) 
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    count +=1
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()