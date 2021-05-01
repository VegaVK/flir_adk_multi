#!/usr/bin/env python
#Modified: April 12, 2021: VVK
# Extract images from rosbag files into .jpeg for training. Bag file names and
#   export paths are hardcoded.
import rospy
import std_msgs
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import numpy as np

def main():
    rospy.init_node('image_extractor', anonymous=True)
    extractor=image_extractor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

class image_extractor:

    def __init__(self):
        self.bridge=CvBridge()
        rospy.Subscriber('/Thermal_Panorama', Image, self.extract)
        self.fileCounter=1

    ## FOR EACH IMAGE:
    # def extract(self,data):
    #     imageFromBag=self.bridge.imgmsg_to_cv2(data, "mono8")
    #     # imageRGB = cv2.cvtColor(imageFromBag,cv2.COLOR_GRAY2RGB)
    #     # The export path is hardcoded:
    #     print(self.fileCounter)
    #     fileNumStr=str(self.fileCounter)
    #     fileNumStr=fileNumStr.zfill(5)
    #     cv2.imwrite('/home/vamsi/RainDataTrain/bag13/'+fileNumStr+'.jpeg',imageFromBag)
    #     self.fileCounter=self.fileCounter+1
    
    ## FOR ALTERNATE IMAGES:
    def extract(self,data):
        imageFromBag=self.bridge.imgmsg_to_cv2(data, "mono8")
        # imageRGB = cv2.cvtColor(imageFromBag,cv2.COLOR_GRAY2RGB)
        # The export path is hardcoded:
        if  (self.fileCounter % 2) == 0:
            print(self.fileCounter)
            fileNumStr=str(self.fileCounter)
            fileNumStr=fileNumStr.zfill(5)
            cv2.imwrite('/home/vamsi/RainDataTrain/bag11/'+fileNumStr+'.jpeg',imageFromBag)
            self.fileCounter=self.fileCounter+1
        else:
            self.fileCounter=self.fileCounter+1




    
if __name__=='__main__':
    main()
#######################

