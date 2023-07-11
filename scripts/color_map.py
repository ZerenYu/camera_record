from turtle import color
import cv2 as cv
import numpy as np
import os
from matplotlib import cm
import time

def color_map(depth_org):
    h, w = depth_org.shape
    temp1 = np.ones((h,w)) * 255
    temp1 = temp1.astype(np.uint8)
    MAXDEPTH = 2
    depth_org.astype(np.float16)
    depth_org = depth_org / 1000
    depth_org[depth_org >= MAXDEPTH] = 0
    depth = depth_org / MAXDEPTH * 255
    depth = depth.astype(np.uint8)
    color_depth = np.dstack((depth, temp1, temp1))
    color_depth = cv.cvtColor(color_depth, cv.COLOR_HSV2BGR)
    return color_depth


if __name__ == "__main__":
    depth_org = cv.imread(os.path.join("/home/yuzeren/CAM/amazon_ws/data/detectmarker6-20230606T203811Z-001/detectmarker6/depth", "depth_1686080963004.png"), cv.IMREAD_UNCHANGED)
    color_depth = color_map(depth_org)
    cv.imwrite(os.path.join("/home/yuzeren/CAM/amazon_ws/data/detectmarker6-20230606T203811Z-001/detectmarker6/depth", "colordepth.png"), color_depth)


    filedir_org = "/home/yuzeren/CAM/amazon_ws/AmazonPackageSamples/good_img1"
    filenames = os.listdir("/home/yuzeren/CAM/amazon_ws/AmazonPackageSamples/good_img1")
    save_dir = "/home/yuzeren/CAM/amazon_ws/AmazonPackageSamples/good_img1_color_map"

    filedir_store = filedir_org
    for filename in filenames:
        if filename[:5] == "depth":
            depth_org =  cv.imread(os.path.join(filedir_org, filename), cv.IMREAD_UNCHANGED)
            color_depth = color_map(depth_org)
            timestamp = int(filename[6:19])
            cv.imwrite(os.path.join(save_dir, "colordepth_{}.png".format(timestamp)), color_depth)

