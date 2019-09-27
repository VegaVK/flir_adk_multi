#!/usr/bin/env python
import rospy
import std_msgs
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import numpy as np

#### BUILT FOR 2 CAMERAS CURRENTLY
def main():
    rospy.init_node('Some_publisher', anonymous=True)
    vsInst=vid_stitch()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

class vid_stitch:

    def __init__(self):
        self.bridge=CvBridge()
        self.image_pub = rospy.Publisher("stitched_image",Image,queue_size=100)
        self.image1=np.array([1])
        self.image2=self.image1
        self.overlapPix=180# number of pixels (horizontal) to average over

        rospy.loginfo('Published stitched image')
        rospy.Subscriber('/flir_boson1/image_rect', Image, self.buildimage1)
        rospy.Subscriber('/flir_boson2/image_rect', Image, self.buildimage2)

    
    def buildimage1(self,data):
        self.image1=self.bridge.imgmsg_to_cv2(data, "mono8")
    def buildimage2(self,data):
        self.image2=self.bridge.imgmsg_to_cv2(data, "mono8")
        self.stitchfun() # Put this in the last buildimage() callback

    def stitchfun(self):
        # For two cameras, basic stitching, average over common pixels
        avgArray=(self.image1[:,640-self.overlapPix:640]+self.image2[:,0:self.overlapPix])/2
        self.Final=np.hstack((self.image1[:,0:640-self.overlapPix],avgArray,self.image2[:,self.overlapPix:640]))
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.Final, "mono8"))
        #print(self.Final.shape)
        
        


if __name__=='__main__':
    main()
    
#    rate = rospy.Rate(10)
#    hello_str = "hello world"
 #   rospy.loginfo(hello_str)
 #   pub.publish(hello_str)
 #   rospy.spin()
 #   exit(0) 