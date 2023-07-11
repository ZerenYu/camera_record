import abc
from threading import local
from manuel_finder import ImagePixelList, mouse_callback
from detect_center_manual import point2pix, pix2point
import cv2
import numpy as np
import os
from matplotlib import cm
import copy
import yaml
import open3d as o3d

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.5
fontColor              = (0,0,255)
thickness              = 2
lineType               = 2


def fit_plane(my_list, depth_intrinsic, depth_img):
    size = my_list.size
    points = np.zeros((size, 3))
    for i, pixelpoint in enumerate(my_list.p_list):
        x = pixelpoint.u
        y = pixelpoint.v
        z = depth_img[y][x]/1000.0
        point = pix2point(x, y, z, depth_intrinsic)
        points[i] = point.T[0][:-1]
        pixelpoint.x, pixelpoint.y, pixelpoint.z = points[i]
    A = copy.deepcopy(points)
    B = A[:, 2]
    A[:, 2] = 1
    abc = np.linalg.inv(points.T @ points) @ points.T @ B
    return abc




if __name__ == "__main__":
    filedir = "/home/yuzeren/CAM/amazon_ws/AmazonPackageSamples"
    store_dir = "/home/yuzeren/CAM/amazon_ws"
    # store_dir = "/Users/YZR/Desktop/CAM/amazon_ws"
    # filedir = "/Users/YZR/Desktop/CAM/amazon_ws/AmazonPackageSamples"
    filename = "rgb_1680636420757.png" 
    depth_filename = "depth_"+filename[4:17]+".png"
    yaml_name = "local.yaml"
    depth_intrinsic  = np.array([[903.5896, 0, 631.955627],
                                 [0, 903.5896, 367.5744],
                                 [0, 0, 1]])
    my_list = ImagePixelList()
    my_list.img = cv2.imread(os.path.join(filedir, filename), cv2.IMREAD_UNCHANGED)


    # my_list.add_point(717, 304)
    # my_list.add_point(183, 312)
    # my_list.add_point(331, 164)
    # my_list.add_point(576, 585)
    # my_list.add_point(703, 508)
    # my_list.add_point(347, 480)
    # my_list.add_point(483, 130)

    # TODO: command out for real manual select points
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback, my_list)
    while(1):
        cv2.imshow('Image', my_list.img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()  
    #####################################################################################

    depth_img = cv2.imread(os.path.join(filedir, depth_filename), cv2.IMREAD_UNCHANGED)
    plane_norm = fit_plane(my_list, depth_intrinsic, depth_img)
    
    # TODO: read write yaml file
    with open(os.path.join(store_dir, yaml_name), 'r') as f:
        data = yaml.safe_load(f)
    if data == None:
        data = {}
    data['plane'] = [float(plane_norm[0]), float(plane_norm[1]), float(plane_norm[2])]
    points_holder = []
    for pixelpoint in my_list.p_list:
        point_holder = []
        temp = [float(pixelpoint.x), float(pixelpoint.y), float(pixelpoint.z)]
        temp1 = [float(pixelpoint.u), float(pixelpoint.v)]
        points_holder.append([temp, temp1])
    data["points"] = points_holder
    with open(os.path.join(store_dir, yaml_name), 'w') as f:
        yaml.dump(data, f)
    #########################################################################################

    # x, y, _ = center_pix.T[0]
    # x, y = int(x), int(y)


    # cv2.namedWindow('Image')
    # cv2.circle(my_list.img,(x, y),2,(0,0,255),-1)
    # cv2.imshow('Image', my_list.img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()  

    cv2.imwrite(os.path.join(store_dir, "center_pose.png"), my_list.img)
