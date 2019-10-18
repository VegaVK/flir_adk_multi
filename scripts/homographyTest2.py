
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
        self.overlapPix=256# number of pixels (horizontal) to average over

        distPlane=6# Distance from camera to plane of image
        trlCam=np.reshape(np.array([0.09,0,0]),(3,1))# translation vector between the two cameras
        normalVec=np.array([0,0,-1])# normal vector of the plane relative to camera
        th=np.deg2rad(20)# rotation about vertical axis, of camera two wrt to camera 1
        R=np.array([[np.cos(th),0, np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]])
        self.homographyMat=R-trlCam*normalVec/distPlane
        self.homographyMat[0,:]=np.array([1,0.00,0])
        self.homographyMat[1,:]=np.array([0.00,1,25])
        self.homographyMat[2,:]=np.array([0.0005,0.00,1])
        
        # Rewriting from Matlab:
        #self.homographyMat=np.array([[-283.7450,59.375,0.299],[26.439,-433.2,0.0444],[-191718.6614,-8935.92,-485.5138]])
        #self.homographyMat=np.array([[-0.00249538,0,0.9863152],[0,-0.0023104,0.16482],[0,0,-0.001437]])
        #self.homographyMat=np.array([[-0.002495386870578,-0.000051197082939,0.986315250396729],[-0.000309730559820,-0.002310410141945,0.164829030632973],[-0.000001565499019,-0.000000242848216,-0.001437022932805]])
        #self.homographyMat=np.linalg.inv(np.transpose(self.homographyMat))
        #self.homographyMat[2,:]=np.array([-0.000000265499019,-0.000000242848216,-0.002237022932805])
        #self.homographyMat[0,:]=np.array([-0.002495386870578,-0.000051197082939,0])
        #self.homographyMat[1,:]=np.array([-0.000309730559820,-0.002310410141945,0])
        #self.homographyMat=np.array([[-0.002495386870578,-0.000051197082939,0.986315250396729],[-0.000309730559820,-0.002310410141945,0.164829030632973],[0,0,-0.001437022932805]])
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
        
        #displacing image 2
        tempOffset=360
        
        #self.Overlap2[:,(self.Overlap2.shape[1]-tempOffset):(self.Overlap2.shape[1])]=np.zeros((512,tempOffset))
        #self.Overlap2[:,tempOffset:(640+tempOffset)]=self.Warped
        #self.Overlap2[:,0:(640-self.overlapPix)]=self.image1[:,0:(640-self.overlapPix)]
        self.warp_pub.publish(self.bridge.cv2_to_imgmsg(self.Warped, "mono8"))

        self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.Overlap1, "mono8"))
        self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.Overlap2, "mono8"))
        self.image_pub3.publish(self.bridge.cv2_to_imgmsg(self.Overlap3, "mono8"))
        self.image_pub2 = rospy.Publisher("stitched_image2",Image,queue_size=100)
        rospy.loginfo('Published warped and stitched image')
        
        
         


if __name__=='__main__':
    main()
#######################

