import cv2
import os

if __name__ == "__main__":
    read_dir = "/home/yuzeren/CAM/amazon_ws/AmazonPackageSamples"
    store_dir = "/home/yuzeren/CAM/amazon_ws/AmazonPackageSamples/resized640360"
    image_list = ["rgb_1680209988712.png", "rgb_1680636420457.png"]

    for img_name in image_list:
        img = cv2.imread(os.path.join(read_dir, img_name))
        resized = cv2.resize(img, (640, 360))
        cv2.imwrite(os.path.join(store_dir, img_name), resized)
