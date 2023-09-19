from tkinter.tix import IMAGE
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from camera_record.srv import capture, captureResponse
import geometry_msgs.msg 
import cv2
import numpy as np
import os
import time
import argparse
from datetime import datetime

# SAVE_DIR = "/home/cam/amazon_ws/data/Three_camera/0005"
SAVE_DIR = "/home/cam/iiwa_ws/recorded_data/exps/images"
bridge = CvBridge()



class LoopManager:
    loop_time = 0 
    img_num = 5
    
    def __init__(self) -> None:
        pass


def rgb_callback(data, time_stamp, id):
    # rospy.loginfo("I heard it height %d, width %d, encoding %s from %d", data.height, data.width, data.encoding, id)
    cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")
    cv2.imwrite(os.path.join(SAVE_DIR, 'rgb'+str(id), '{}.png'.format(time_stamp)), cv2_img)

def depth_callback(data, time_stamp, id):
    # rospy.loginfo("I heard it height %d, width %d, encoding %s", data.height, data.width, data.encoding)
    cv2_img = bridge.imgmsg_to_cv2(data, "16UC1")
    # rospy.loginfo("max: %d, min: %d, avg: %d", np.max(cv2_img), np.min(cv2_img), np.average(cv2_img))
    cv2.imwrite(os.path.join(SAVE_DIR, 'depth'+str(id), '{}.png'.format(time_stamp)), cv2_img, (cv2.IMWRITE_PXM_BINARY, 0))

    
def listener(req):
    print("Heard request")
    prefix = req.prefix
    os.makedirs(SAVE_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(SAVE_DIR, "rgb1")):
        os.mkdir(os.path.join(SAVE_DIR, "rgb1"))
    if not os.path.exists(os.path.join(SAVE_DIR, "depth1")):
        os.mkdir(os.path.join(SAVE_DIR, "depth1"))
    if not os.path.exists(os.path.join(SAVE_DIR, "rgb2")):
        os.mkdir(os.path.join(SAVE_DIR, "rgb2"))
    if not os.path.exists(os.path.join(SAVE_DIR, "depth2")):
        os.mkdir(os.path.join(SAVE_DIR, "depth2"))
    if not os.path.exists(os.path.join(SAVE_DIR, "rgb3")):
        os.mkdir(os.path.join(SAVE_DIR, "rgb3"))
    if not os.path.exists(os.path.join(SAVE_DIR, "depth3")):
        os.mkdir(os.path.join(SAVE_DIR, "depth3"))
    # while manager.loop_time < manager.img_num:
    #     manager.loop_time += 1
    # while(1):
    now = datetime.now()
    time_stamp = now.strftime("%m-%d-%Y-%H-%M-%S")
    wholename = prefix
    rgb1 = rospy.wait_for_message("/camera1/color/image_raw", Image, timeout=None)
    rgb_callback(rgb1, wholename, 1)
    depth1 = rospy.wait_for_message("/camera1/aligned_depth_to_color/image_raw", Image, timeout=None)
    depth_callback(depth1, wholename,1)
    rgb2 = rospy.wait_for_message("/camera2/color/image_raw", Image, timeout=None)
    rgb_callback(rgb2, wholename, 2)
    depth2 = rospy.wait_for_message("/camera2/aligned_depth_to_color/image_raw", Image, timeout=None)
    depth_callback(depth2, wholename, 2)
    rgb3 = rospy.wait_for_message("/camera3/color/image_raw", Image, timeout=None)
    rgb_callback(rgb3, wholename, 3)
    depth3 = rospy.wait_for_message("/camera3/aligned_depth_to_color/image_raw", Image, timeout=None)
    depth_callback(depth3, wholename, 3)

    print("{}".format(wholename))
    return captureResponse(wholename)
    # rospy.Subscriber("/camera/color/image_raw", Image, rgb_callback)
    # rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_callback)
    
    # if manager.loop_time < manager.img_num:
    #     manager.loop_time += 1
    #     rospy.spin()
    # else:
    #     exit(0)

if __name__ == '__main__':
    rospy.init_node('Recorder', anonymous=True)
    s = rospy.Service('three_camera', capture, listener)
    rospy.loginfo("camera record started")
    rospy.spin()

