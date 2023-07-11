import cv2
import numpy as np
import os
from matplotlib import cm

class PixelVoxel:
    u = 0
    v = 0
    x = 0
    y = 0
    z = 0
    def __init__(self) -> None:
        pass
    def set_uv(self, x, y):
        self.u = x
        self.v = y
    def set_pose(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    

class ImagePixelList:
    p_list = []
    img = None
    size = 0
    def __init__(self) -> None:
        pass
    def add_point(self, x, y):
        temp = PixelVoxel()
        temp.set_uv(x, y)
        self.p_list.append(temp)
        self.size += 1
        x = 0

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print('Pixel value at ({}, {}): {}'.format(x, y, param.img[y, x]))
        cv2.circle(param.img,(x,y),2,(0,0,255),-1)
        param.add_point(x, y)
        

if __name__ == "__main__":
    # filedir = "/home/yuzeren/CAM/amazon_ws/AmazonPackageSamples"
    filedir = "/Users/YZR/Desktop/CAM/amazon_ws/AmazonPackageSamples"
    filename = "rgb_1680209988712.png"
    my_list = ImagePixelList()
    my_list.img = cv2.imread(os.path.join(filedir, filename), cv2.IMREAD_UNCHANGED)
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback, my_list)
    while(1):
        cv2.imshow('Image', my_list.img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()  