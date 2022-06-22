
import cv2 as cv
import shutil
import os
from pathlib import Path
from transforms import *



__author__ = "Alberto Abarzua"

FRAME_WIDTH = 32


def run_recording():
    """Function used to record frames from webcam.
    """
    BASE = Path(__file__).parent.parent.joinpath("data")
    DATA = BASE / "recording_data"
    if os.path.exists(DATA): 
        shutil.rmtree(DATA)

    os.mkdir(DATA) 

    vid = cv.VideoCapture(0)
    count =0
    while(True):
        succes, pre_frame = vid.read()
        
        frame = prep_transform(pre_frame,des_width = FRAME_WIDTH)
        cv.imshow('Recording', rescaleFrame(frame,8))
        cv.imwrite(DATA.joinpath("frame{}.jpg".format(count)).__str__(), frame) 
        
        count +=1
        if cv.waitKey(1) & 0xFF == 27:
            break
    vid.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    run_recording()