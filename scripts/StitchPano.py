#!/usr/bin/env python
import rospy
import std_msgs
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import numpy as np

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
        self.PanoPub = rospy.Publisher("Thermal_Panorama",Image,queue_size=100)
        self.TempWarpPub=rospy.Publisher("Warp1_2",Image,queue_size=100)
        self.overlapPix12=190#180,160,200,200,
        self.overlapPix23=187
        self.overlapPix34=215
        self.overlapPix45=220 # Guess, needs to be tuned
        self.image1Stat=[]
        self.image2Stat=[]
        self.image3Stat=[]
        self.image4Stat=[]
        self.image5Stat=[]
        
        self.smoothingPix=10# Number of pixels to smooth over
        # Create gradient arrays for Left and Right
        self.gradientArrLeft=np.linspace(1.0,0.0, self.smoothingPix, endpoint=True)
        self.gradientArrLeft=np.tile(self.gradientArrLeft,(512,1))
        self.gradientArrRight=np.linspace(0.0,1.0, self.smoothingPix, endpoint=True)
        self.gradientArrRight=np.tile(self.gradientArrRight,(512,1))

        self.homographyMat1_2=np.array([[1,0,0],[0,1,0],[0.00015,0.00,0.92]])
        self.homographyMat2_3=np.array([[1,0,0],[0,1,0],[0.00006,0.00,0.94]])
        self.homographyMat3_4=np.array([[1,0,0],[0,1,0],[-0.00012,0.00,1]])
        self.homographyMat4_5=np.array([[1,0,0],[0,1,0],[-0.00021,0.00,1]]) # Guess, needs to be tuned
        
        rospy.Subscriber('/flir_boson1/image_rect', Image, self.buildimage1)
        rospy.Subscriber('/flir_boson2/image_rect', Image, self.buildimage2)
        rospy.Subscriber('/flir_boson3/image_rect', Image, self.buildimage3)
        rospy.Subscriber('/flir_boson4/image_rect', Image, self.buildimage4)
        rospy.Subscriber('/flir_boson5/image_rect', Image, self.buildimage5)
         
    
    def buildimage1(self,data):
        self.image1=self.bridge.imgmsg_to_cv2(data, "mono8")
        # self.image1Stat=1

    def buildimage2(self,data):
        self.image2=self.bridge.imgmsg_to_cv2(data, "mono8")
        # self.image2Stat=1

    def buildimage3(self,data):
        self.image3=self.bridge.imgmsg_to_cv2(data, "mono8")
        # self.image3Stat=1

    def buildimage4(self,data):
        self.image4=self.bridge.imgmsg_to_cv2(data, "mono8")
        # self.image4Stat=1
        
    def buildimage5(self,data):
        self.image5=self.bridge.imgmsg_to_cv2(data, "mono8")
        # self.image5Stat=1
        self.stitchfun() # Put this in the LAST buildimage() callback 
   
    def stitchfun(self):
        # First convert Cam1 and Cam5 to Cam2 and Cam4 frames respectively
        self.Warped5 = cv2.warpPerspective(self.image5,self.homographyMat4_5,(640,512))
        self.Warped1 = cv2.warpPerspective(self.image1,self.homographyMat1_2,(640,512))
        SmoothingArray1_2=np.multiply(self.Warped1[:,-(self.smoothingPix+self.overlapPix12):-self.overlapPix12],self.gradientArrLeft)+np.multiply(self.image2[:,0:self.smoothingPix],self.gradientArrRight)
        SmoothingArray4_5=np.multiply(self.image4[:,-self.smoothingPix:],self.gradientArrLeft)+np.multiply(self.Warped5[:,self.overlapPix45:self.overlapPix45+self.smoothingPix],self.gradientArrRight)
        #Create combined Cam1-2 and Cam 4-5 images
        self.Warped4_5=np.hstack((self.image4[:,0:-(self.smoothingPix)],np.uint8(np.round(SmoothingArray4_5,0)),self.Warped5[:,(self.smoothingPix+self.overlapPix45):]))
        self.Warped1_2=np.hstack((self.Warped1[:,0:-(self.smoothingPix+self.overlapPix12)],np.uint8(np.round(SmoothingArray1_2,0)),self.image2[:,self.smoothingPix:]))
        #Now warp these to Cam3 Frame
        self.Warped3_45=cv2.warpPerspective(self.Warped4_5,self.homographyMat3_4,((2*640-self.overlapPix34),512))
        self.Warped12_3=cv2.warpPerspective(self.Warped1_2,self.homographyMat2_3,((2*640-self.overlapPix23),512))

        SmoothingArray12_3=np.multiply(self.Warped12_3[:,-(self.smoothingPix+self.overlapPix23):-self.overlapPix23],self.gradientArrLeft)+np.multiply(self.image3[:,0:self.smoothingPix],self.gradientArrRight)
        SmoothingArray3_45=np.multiply(self.image3[:,-self.smoothingPix:],self.gradientArrLeft)+np.multiply(self.Warped3_45[:,self.overlapPix34:self.overlapPix34+self.smoothingPix],self.gradientArrRight)
        #Stitch all of them together
        self.Panorama=np.hstack((self.Warped12_3[:,0:-(self.overlapPix23+self.smoothingPix)],np.uint8(np.round(SmoothingArray12_3,0)),self.image3[:,self.smoothingPix:-(self.smoothingPix)],np.uint8(np.round(SmoothingArray3_45,0)),self.Warped3_45[:,(self.smoothingPix+self.overlapPix34):]))
        # Conversion to RGB
        #print(self.Panorama.shape)
        self.Panorama=cv2.cvtColor(self.Panorama,cv2.COLOR_GRAY2RGB)                
        # Publishers
        self.image1Stat=0
        self.image2Stat=0
        self.image3Stat=0
        self.image4Stat=0
        self.image5Stat=0
        self.PanoPub.publish(self.bridge.cv2_to_imgmsg(self.Panorama, "rgb8"))
        self.TempWarpPub.publish(self.bridge.cv2_to_imgmsg(self.Warped1_2, "mono8"))
        rospy.loginfo('Published Panorama')

        
         


if __name__=='__main__':
    main()
#######################

