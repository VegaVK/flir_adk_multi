
#!/usr/bin/env python
import rospy
import std_msgs
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import numpy as np
#### BUILT FOR 2 CAMERAS CURRENTLY
## Orientation of camera frame is selected as per Wikipedia article on Homography (Computer Vision)
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
        self.image_pub1 = rospy.Publisher("stitched_image1",Image,queue_size=100)
        self.image_pub2 = rospy.Publisher("stitched_image2",Image,queue_size=100)
        self.image_pub3 = rospy.Publisher("stitched_image3",Image,queue_size=100)
        self.warp_pub=rospy.Publisher("warped_image",Image,queue_size=100)

        self.overlapPix=1# number of pixels (horizontal) to average over

        self.homographyMat=np.zeros((3,4))
        self.homographyMat[0,:]=np.array([1,0.00,0,0])
        self.homographyMat[1,:]=np.array([0.00,1,22,0])
        self.homographyMat[2,:]=np.array([0.0001,0.00,1,0])
       
        print(self.homographyMat)#-0.000309730559820
        
        rospy.Subscriber('/flir_boson3/image_rect', Image, self.buildimage1)
        rospy.Subscriber('/flir_boson4/image_rect', Image, self.buildimage2)

    
    def buildimage1(self,data):
        self.image1=self.bridge.imgmsg_to_cv2(data, "mono8")
    def buildimage2(self,data):
        self.image2=self.bridge.imgmsg_to_cv2(data, "mono8")
        self.stitchfun() # Put this in the last buildimage() callback

    def stitchfun(self):

        self.Warped = cv2.warpPerspective(self.image2,self.homographyMat,(self.image2.shape[1],self.image2.shape[0]))
        self.Overlap1=np.hstack((self.image1,self.image2[:,(self.overlapPix):640]))
        self.Overlap2=np.hstack((self.image1,self.Warped))
        self.Overlap3=np.hstack((self.image1,self.Warped[:,(self.overlapPix):640]))
        self.warp_pub.publish(self.bridge.cv2_to_imgmsg(self.Warped, "mono8"))

        self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.Overlap1, "mono8"))
        self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.Overlap2, "mono8"))
        self.image_pub3.publish(self.bridge.cv2_to_imgmsg(self.Overlap3, "mono8"))
        rospy.loginfo('Published warped and stitched image')
        
        
         


if __name__=='__main__':
    main()
#######################

