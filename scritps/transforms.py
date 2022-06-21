import math
import cv2 as cv

def resize_transform(frame,des_width):

    scale = des_width/frame.shape[0]
    dims = (int(frame.shape[1]*scale),int(frame.shape[0]*scale))
    return cv.resize(frame,dims,interpolation= cv.INTER_AREA )

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

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

from torch.utils.data import Dataset, DataLoader,random_split
# class ObjectLoader(Dataset):
#     def __init__(self,data,transforms):
#         super(ObjectLoader,self).__init__()
#         self.data = data
#         self.transforms = transforms
#         self.transform_data()
#     def transform_data(self):
#         self.data
#         for elem in self.data:
#             for tr in self.transforms:
#                 print("elem: ",elem)
#                 elem = tr(elem[0])


#     def __getitem__(self,idx):
#         img ,label= self.data[idx][0], self.data[idx][1]
#         return img,label
#     def __len__(self):
#         return len(self.data)