from tkinter.tix import IMAGE
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
import time

SAVE_DIR = "/home/cam/amazon_ws/data/calibration/0014"
bridge = CvBridge()


class LoopManager:
    loop_time = 0 
    img_num = 5
    
    def __init__(self) -> None:
        pass


def rgb_callback(data, time_stamp):
    rospy.loginfo("I heard it height %d, width %d, encoding %s", data.height, data.width, data.encoding)
    cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")
    cv2.imwrite(os.path.join(SAVE_DIR, 'rgb', '{}.png'.format(time_stamp)), cv2_img)

def depth_callback(data, time_stamp):
    # rospy.loginfo("I heard it height %d, width %d, encoding %s", data.height, data.width, data.encoding)
    cv2_img = bridge.imgmsg_to_cv2(data, "16UC1")
    rospy.loginfo("max: %d, min: %d, avg: %d", np.max(cv2_img), np.min(cv2_img), np.average(cv2_img))
    cv2.imwrite(os.path.join(SAVE_DIR, 'depth', '{}.png'.format(time_stamp)), cv2_img, (cv2.IMWRITE_PXM_BINARY, 0))

    
def listener(manager):

    rospy.init_node('Recorder', anonymous=True)
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if not os.path.exists(os.path.join(SAVE_DIR, "rgb")):
        os.mkdir(os.path.join(SAVE_DIR, "rgb"))
    if not os.path.exists(os.path.join(SAVE_DIR, "depth")):
        os.mkdir(os.path.join(SAVE_DIR, "depth"))
    # while manager.loop_time < manager.img_num:
    #     manager.loop_time += 1
    while(1):
        time_stamp = int(time.time()*1000)
        rgb_msg = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=None)
        rgb_callback(rgb_msg, time_stamp)
        depth_msg = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image, timeout=None)
        depth_callback(depth_msg, time_stamp)


    # rospy.Subscriber("/camera/color/image_raw", Image, rgb_callback)
    # rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_callback)
    
    # if manager.loop_time < manager.img_num:
    #     manager.loop_time += 1
    #     rospy.spin()
    # else:
    #     exit(0)

if __name__ == '__main__':
    my_lm = LoopManager()
    listener(my_lm)
