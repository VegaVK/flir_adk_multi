
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
        self.image_pub1 = rospy.Publisher("DirectStitch",Image,queue_size=100)
        self.image_pub2 = rospy.Publisher("WarpStitch1",Image,queue_size=100)
        self.warp_pub=rospy.Publisher("warped_image",Image,queue_size=100)
        self.overlapPix=273
        self.smoothingPix=10 # Number of pixels to smooth over
        # Create gradient arrays for image 1 and image 2
        self.gradientArr1=np.linspace(1.0,0.0, self.smoothingPix, endpoint=True)
        self.gradientArr1=np.tile(self.gradientArr1,(512,1))
        self.gradientArr2=np.linspace(0.0,1.0, self.smoothingPix, endpoint=True)
        self.gradientArr2=np.tile(self.gradientArr2,(512,1))
        self.homographyMat1=np.array([[1,0,0],[0,1,-6],[-0.00018,0.00,1]])
        rospy.Subscriber('/flir_boson3/image_rect', Image, self.buildimage1)
        rospy.Subscriber('/flir_boson4/image_rect', Image, self.buildimage2)

    
    def buildimage1(self,data):
        self.image1=self.bridge.imgmsg_to_cv2(data, "mono8")
    def buildimage2(self,data):
        self.image2=self.bridge.imgmsg_to_cv2(data, "mono8")
        self.stitchfun() # Put this in the last buildimage() callback

    def stitchfun(self):
        self.Warped = cv2.warpPerspective(self.image2,self.homographyMat1,(self.image2.shape[1],self.image2.shape[0]))
        self.Overlap1=np.hstack((self.image1,self.image2[:,(self.overlapPix):640]))
        SmoothingArray1=np.multiply(self.image1[:,640-self.smoothingPix:640],self.gradientArr1)+np.multiply(self.Warped[:,self.overlapPix:self.overlapPix+self.smoothingPix],self.gradientArr2)
        self.Overlap2=np.hstack((self.image1[:,0:640-(self.smoothingPix)],np.uint8(np.round(SmoothingArray1,0)),self.Warped[:,(self.smoothingPix+self.overlapPix):640]))
        
        # Publishers
        self.warp_pub.publish(self.bridge.cv2_to_imgmsg(self.Warped, "mono8"))
        self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.Overlap1, "mono8"))
        self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.Overlap2, "mono8"))
        rospy.loginfo('Published warped and stitched image')
        
        
         


if __name__=='__main__':
    main()
#######################

