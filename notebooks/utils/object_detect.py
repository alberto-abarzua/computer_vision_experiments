from pickle import FALSE
from turtle import back
import cv2
import numpy as np
from transforms import *
import os
from pathlib import Path
import shutil
import torch
import sys
import torchvision.transforms as transforms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import denseNet as dn
__author__ = "Alberto Abarzua"



BASE = Path(__file__).parent.parent.joinpath("data")

def run_obj_detection():
    """Used to run the object detection script using webcam as video input.
    """
    
    DATA = BASE / "obj_log"
    DATAROIS = DATA/"rois"

    SAVE_ROIS = False
    PRESSED = False
    dirs = [str(x) for x in range(1,10)]
    CUR_DIR = "1"

    if SAVE_ROIS:
        if os.path.exists(DATAROIS): 
            shutil.rmtree(DATAROIS)

        os.mkdir(DATAROIS) 
        for elem in dirs:
            os.mkdir(DATAROIS/elem)

    MODEL = BASE.joinpath("dataset/output_data")


    num_frames = 0
    background = None
    FRAME_SIZE = 256
    FRAMES_CALIBRATION = 30
    THRESH = 50
    model_data = torch.load(MODEL/"densenet")
    label_dict = model_data["labels"]
    model = dn.DenseNet(len(label_dict))
    model.load_state_dict(model_data["model_state"])
    trs= [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
    idx_label = {v: k for k, v in label_dict.items()}
    print("Labels:\n",label_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    
    cam = cv2.VideoCapture(0)
    count =0
    while True:
        ret, pre_frame = cam.read()
        roi = None
        frame = prep_transform(pre_frame,des_width = FRAME_SIZE)
        frame_save_copy = frame.copy()
        frame_copy = frame.copy()
    
        if num_frames < FRAMES_CALIBRATION:
            background = define_background(background, frame_copy)
            if num_frames <= FRAMES_CALIBRATION-2:
                cv2.putText(frame, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("obj", rescaleFrame(frame,5))
            else:
                print("SAVING Bg!")
                background = end_background(background)
                np.save(DATA/"background",background)
            num_frames += 1
            continue
                
        
        val = get_objs(background,frame_copy,THRESH,200)
        if val is not None: 
            objs,_= val
            for elem in objs:
                (x,y,w,h) = elem
                roi = get_roi(frame_save_copy,elem,FRAME_SIZE)
                roi_rgb = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
                img = dn.apply_transforms(trs,[roi_rgb]).to(device)
                arg = torch.argmax(model.predict_proba(img)[0]).item()
                pred = idx_label[arg]
                
                if SAVE_ROIS and PRESSED:
                    cv2.imwrite(DATAROIS.joinpath(CUR_DIR,"roi{}.jpg".format(count)).__str__(), roi) 
                    count +=1
                    
                cv2.imshow("obj1", rescaleFrame(roi,5))
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
                cv2.putText(frame,pred, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
        PRESSED = False
        num_frames += 1

        cv2.imshow("obj", rescaleFrame(frame,5))


        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

        if k == ord("r"): #Save current frame to obj_log:
            np.save(DATA/"frame",frame_save_copy)
            if roi is not None:
                np.save(DATA/"roi",roi)
        if k == ord("b"):
            PRESSED= True
        if k>0 and chr(k).isdigit():
            CUR_DIR= str(chr(k))

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  
    run_obj_detection()