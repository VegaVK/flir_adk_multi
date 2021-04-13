#!/usr/bin/env python
#Modified: March 17, 2021: VVK
#
import rospy
import std_msgs
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import numpy as np

def main():
    rospy.init_node('Panorama', anonymous=True)
    vsInst=vid_stitch()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

class vid_stitch:

    def __init__(self):
        self.bridge=CvBridge()
        self.PanoPub = rospy.Publisher("Thermal_Panorama",Image,queue_size=100)
        # self.TempWarpPub=rospy.Publisher("Warp1_2",Image,queue_size=100)
        self.overlapPix12=170#180,160,200,200,
        self.overlapPix23=175
        self.overlapPix34=185
        self.overlapPix45=180 # Guess, needs to be tuned
        
        self.smoothingPix=10# Number of pixels to smooth over
        # Create gradient arrays for Left and Right
        self.gradientArrLeft=np.linspace(1.0,0.0, self.smoothingPix, endpoint=True)
        self.gradientArrLeft=np.tile(self.gradientArrLeft,(512,1))

        self.gradientArrRight=np.linspace(0.0,1.0, self.smoothingPix, endpoint=True)
        self.gradientArrRight=np.tile(self.gradientArrRight,(512,1))

        self.homographyMat1_2=np.array([[1,0,0],[0,1,0],[0.00018,0.00,0.92]])
        self.homographyMat2_3=np.array([[1,0,0],[0,1,0],[0.00006,0.00,0.94]])
        self.homographyMat3_4=np.array([[1,0,0],[0,1,0],[-0.00006,0.00,1.0]])
        self.homographyMat4_5=np.array([[1,0,0],[0,1,0],[-0.00019,0.00,1.02]]) # Guess, needs to be tuned
        
        rospy.Subscriber('/flir_boson1/image_rect', Image, self.buildimage1)
        rospy.Subscriber('/flir_boson2/image_rect', Image, self.buildimage2)
        rospy.Subscriber('/flir_boson3/image_rect', Image, self.buildimage3)
        rospy.Subscriber('/flir_boson4/image_rect', Image, self.buildimage4)
        rospy.Subscriber('/flir_boson5/image_rect', Image, self.buildimage5)
         
    
    def buildimage1(self,data):
        self.image1=self.bridge.imgmsg_to_cv2(data, "rgb8")
        # self.image1Stat=1

    def buildimage2(self,data):
        self.image2=self.bridge.imgmsg_to_cv2(data, "rgb8")
        # self.image2Stat=1

    def buildimage3(self,data):
        self.image3=self.bridge.imgmsg_to_cv2(data, "rgb8")
        # self.image3Stat=1

    def buildimage4(self,data):
        self.image4=self.bridge.imgmsg_to_cv2(data, "rgb8")
        # self.image4Stat=1
        
    def buildimage5(self,data):
        self.image5=self.bridge.imgmsg_to_cv2(data, "rgb8")
        # self.image5Stat=1
        self.stitchfun() # Put this in the LAST buildimage() callback 
   
    def stitchfun(self):
        # First convert Cam1 and Cam5 to Cam2 and Cam4 frames respectively
        self.Warped5 = cv2.warpPerspective(self.image5,self.homographyMat4_5,(640,512))
        self.Warped1 = cv2.warpPerspective(self.image1,self.homographyMat1_2,(640,512))
        SmoothingArray1_2C1=np.uint8(np.multiply(self.Warped1[:,-(self.smoothingPix+self.overlapPix12):-self.overlapPix12,0],self.gradientArrLeft)+np.multiply(self.image2[:,0:self.smoothingPix,0],self.gradientArrRight))
        SmoothingArray1_2C2=np.uint8(np.multiply(self.Warped1[:,-(self.smoothingPix+self.overlapPix12):-self.overlapPix12,1],self.gradientArrLeft)+np.multiply(self.image2[:,0:self.smoothingPix,1],self.gradientArrRight))
        SmoothingArray1_2C3=np.uint8(np.multiply(self.Warped1[:,-(self.smoothingPix+self.overlapPix12):-self.overlapPix12,2],self.gradientArrLeft)+np.multiply(self.image2[:,0:self.smoothingPix,2],self.gradientArrRight))
        SmoothingArray1_2=cv2.merge([SmoothingArray1_2C1,SmoothingArray1_2C2,SmoothingArray1_2C3])
        SmoothingArray4_5C1=np.uint8(np.multiply(self.image4[:,-self.smoothingPix:,0],self.gradientArrLeft)+np.multiply(self.Warped5[:,self.overlapPix45:self.overlapPix45+self.smoothingPix,0],self.gradientArrRight))
        SmoothingArray4_5C2=np.uint8(np.multiply(self.image4[:,-self.smoothingPix:,1],self.gradientArrLeft)+np.multiply(self.Warped5[:,self.overlapPix45:self.overlapPix45+self.smoothingPix,1],self.gradientArrRight))
        SmoothingArray4_5C3=np.uint8(np.multiply(self.image4[:,-self.smoothingPix:,2],self.gradientArrLeft)+np.multiply(self.Warped5[:,self.overlapPix45:self.overlapPix45+self.smoothingPix,2],self.gradientArrRight))
        SmoothingArray4_5=cv2.merge([SmoothingArray4_5C1,SmoothingArray4_5C2,SmoothingArray4_5C3])


        # #Create combined Cam1-2 and Cam 4-5 images
        self.Warped4_5=np.hstack((self.image4[:,0:-(self.smoothingPix)],SmoothingArray4_5,self.Warped5[:,(self.smoothingPix+self.overlapPix45):]))
        self.Warped1_2=np.hstack((self.Warped1[:,0:-(self.smoothingPix+self.overlapPix12)],SmoothingArray1_2,self.image2[:,self.smoothingPix:]))
        # #Now warp these to Cam3 Frame
        self.Warped3_45=cv2.warpPerspective(self.Warped4_5,self.homographyMat3_4,((2*640-self.overlapPix34),512))
        self.Warped12_3=cv2.warpPerspective(self.Warped1_2,self.homographyMat2_3,((2*640-self.overlapPix23),512))

        SmoothingArray12_3C1=np.uint8(np.multiply(self.Warped12_3[:,-(self.smoothingPix+self.overlapPix23):-self.overlapPix23,0],self.gradientArrLeft)+np.multiply(self.image3[:,0:self.smoothingPix,0],self.gradientArrRight))        
        SmoothingArray12_3C2=np.uint8(np.multiply(self.Warped12_3[:,-(self.smoothingPix+self.overlapPix23):-self.overlapPix23,1],self.gradientArrLeft)+np.multiply(self.image3[:,0:self.smoothingPix,1],self.gradientArrRight))        
        SmoothingArray12_3C3=np.uint8(np.multiply(self.Warped12_3[:,-(self.smoothingPix+self.overlapPix23):-self.overlapPix23,2],self.gradientArrLeft)+np.multiply(self.image3[:,0:self.smoothingPix,2],self.gradientArrRight))        
        SmoothingArray12_3=cv2.merge([SmoothingArray12_3C1,SmoothingArray12_3C2,SmoothingArray12_3C3])

        SmoothingArray3_45C1=np.uint8(np.multiply(self.image3[:,-self.smoothingPix:,0],self.gradientArrLeft)+np.multiply(self.Warped3_45[:,self.overlapPix34:self.overlapPix34+self.smoothingPix,0],self.gradientArrRight))
        SmoothingArray3_45C2=np.uint8(np.multiply(self.image3[:,-self.smoothingPix:,1],self.gradientArrLeft)+np.multiply(self.Warped3_45[:,self.overlapPix34:self.overlapPix34+self.smoothingPix,1],self.gradientArrRight))
        SmoothingArray3_45C3=np.uint8(np.multiply(self.image3[:,-self.smoothingPix:,2],self.gradientArrLeft)+np.multiply(self.Warped3_45[:,self.overlapPix34:self.overlapPix34+self.smoothingPix,2],self.gradientArrRight))
        SmoothingArray3_45=cv2.merge([SmoothingArray3_45C1,SmoothingArray3_45C2,SmoothingArray3_45C3])
        
        # #Stitch all of them together
        self.Panorama=np.hstack((self.Warped12_3[:,0:-(self.overlapPix23+self.smoothingPix)],SmoothingArray12_3,self.image3[:,self.smoothingPix:-(self.smoothingPix)],SmoothingArray3_45,self.Warped3_45[:,(self.smoothingPix+self.overlapPix34):]))
        # # Conversion to RGB
        # #print(self.Panorama.shape)
        # # self.Panorama=cv2.cvtColor(self.Panorama,cv2.COLOR_GRAY2RGB)                
        # # Publishers
        # self.Warped1 = cv2.warpPerspective(self.image1,self.homographyMat1_2,(640,512,3))
        # SmoothingArray4_5=cv2.cvtColor(SmoothingArray4_5, cv2.COLOR_GRAY2RGB)
        # SmoothingArray4_5=cv2.convertScaleAbs(SmoothingArray4_5, cv2.CV_8UC3, 1, 0)
        # SmoothingArray4_5=cv2.cvtColor(SmoothingArray4_5,cv2.COLOR_GRAY2RGB)   
        # print(SmoothingArray4_5.shape)
        # self.Panorama=np.hstack(SmoothingArray4_5)
        # self.Panorama=cv2.cvtColor(self.Panorama,cv2.COLOR_RGB2GRAY)  
        # self.PanoPub.publish(self.bridge.cv2_to_imgmsg(self.Panorama, "mono8"))
        # self.Panorama=cv2.cvtColor(self.Panorama,cv2.COLOR_RGB2GRAY)  
        self.PanoPub.publish(self.bridge.cv2_to_imgmsg(self.Panorama, "rgb8"))
        # self.TempWarpPub.publish(self.bridge.cv2_to_imgmsg(self.Warped1_2, "mono8"))
        rospy.loginfo('Published Panorama')

        
         


if __name__=='__main__':
    main()
#######################

