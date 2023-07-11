from threading import local
from manuel_finder import ImagePixelList, mouse_callback
import cv2
import numpy as np
import os
from matplotlib import cm
import copy
import yaml

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.5
fontColor              = (0,0,255)
thickness              = 2
lineType               = 2

def pix2point(x, y, z, intrinsic):
    inverse_intrinsic = np.array([[1/intrinsic[0][0], 0, 0, 0],
                                  [0, 1/intrinsic[1][1], 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    local_pose = np.ones((4, 1))
    local_pose[0][0] = x-intrinsic[0][2]
    local_pose[1][0] = y-intrinsic[1][2]
    local_pose[2][0] = 1
    local_pose[3][0] = 1/z
    pose = z * inverse_intrinsic @ local_pose
    return pose


def points_center(pixel_list, depth_img, intrinsic):
    inverse_intrinsic = np.array([[1/intrinsic[0][0], 0, 0, 0],
                                  [0, 1/intrinsic[1][1], 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    world_pose = []
    for pixel in pixel_list.p_list:
        x = pixel.u
        y = pixel.v
        z = depth_img[y][x]/1000.0
        pose = pix2point(x, y, z, intrinsic)
        world_pose.append(pose)
        pixel.x, pixel.y, pixel.z, _ = pose.T[0]
        cv2.putText(my_list.img,"x: {}, y: {}, z: {}".format(round(pixel.x, 3), 
                                                         round(pixel.y, 3), 
                                                         round(pixel.z, 3)), 
        (x - 100, y - 10), 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    world_pose = np.array(world_pose)
    center_pose = np.average(world_pose, axis = 0)
    return center_pose

def point2pix(point, intrinsic):
    result = np.ones((3, 1))
    local_intrinsic = copy.deepcopy(intrinsic)
    local_intrinsic[0][2] = 0
    local_intrinsic[1][2] = 0
    if point.shape == (4, 1):
        result = local_intrinsic @ point[0:3] / point[2][0]
    
    return result


if __name__ == "__main__":
    filedir = "/home/yuzeren/CAM/amazon_ws/AmazonPackageSamples"
    # only change filename to specify which img to use
    filename = "rgb_1680209988712.png" 
    store_dir = "/home/yuzeren/CAM/amazon_ws"
    depth_filename = "depth_"+filename[4:17]+".png"
    depth_intrinsic  = np.array([[903.5896, 0, 631.955627],
                                 [0, 903.5896, 367.5744],
                                 [0, 0, 1]])
    my_list = ImagePixelList()
    my_list.img = cv2.imread(os.path.join(filedir, filename), cv2.IMREAD_UNCHANGED)

    # my_list.add_point(749, 378)
    # my_list.add_point(520, 561)
    # my_list.add_point(218, 260)
    # my_list.add_point(460, 158)

    # TODO: command out for real manual select points
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback, my_list)
    while(1):
        cv2.imshow('Image', my_list.img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()  

    depth_img = cv2.imread(os.path.join(filedir, depth_filename), cv2.IMREAD_UNCHANGED)
    center_pose = points_center(my_list, depth_img, depth_intrinsic)
    center_pix = point2pix(center_pose, depth_intrinsic)
    x, y, _ = center_pix.T[0]
    x, y = int(x), int(y)

    cv2.putText(my_list.img,"x: {}, y: {}, z: {}".format(round(center_pose[0][0], 3), 
                                                         round(center_pose[1][0], 3), 
                                                         round(center_pose[2][0], 3)), 
        (x - 100, y - 10), 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)

    cv2.namedWindow('Image')
    cv2.circle(my_list.img,(x, y),2,(0,0,255),-1)
    cv2.imshow('Image', my_list.img)
    cv2.waitKey()
    cv2.destroyAllWindows()  

    cv2.imwrite(os.path.join(store_dir, "center_pose.png"), my_list.img)
