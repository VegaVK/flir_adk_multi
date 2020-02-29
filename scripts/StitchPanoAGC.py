
#!/usr/bin/env python
# ----------------ONLY use this for stitching in RAW16 mode--------------------
# Works most of the time but isn't robust. Probably better to re-write driver if we want proper AGC


import rospy
import std_msgs 
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import numpy as np
#### BUILT FOR 5 Cameras ( #5 is currently commented out)

def main():
    rospy.init_node('Some_publisher', anonymous=True)
    vsInst=vid_stitch()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

class vid_stitch:
     # Store AGC gains:
    def MinAgc1(self,data):
        self.MinAgcVal1=data.data
    def MinAgc2(self,data):
        self.MinAgcVal2=data.data
    def MinAgc3(self,data):
        self.MinAgcVal3=data.data
    def MinAgc4(self,data):
        self.MinAgcVal4=data.data
    def MinAgc5(self,data):
        self.MinAgcVal5=data.data

    def MaxAgc1(self,data):
        self.MaxAgcVal1=data.data
    def MaxAgc2(self,data):
        self.MaxAgcVal2=data.data
    def MaxAgc3(self,data):
        self.MaxAgcVal3=data.data
    def MaxAgc4(self,data):
        self.MaxAgcVal4=data.data
    def MaxAgc5(self,data):
        self.MaxAgcVal5=data.data

    def __init__(self):
        self.bridge=CvBridge()
        self.PanoPub = rospy.Publisher("Thermal_Panorama",Image,queue_size=100)
        #self.TempWarpPub=rospy.Publisher("Warp1_2",Image,queue_size=100) # For Debugging
        self.overlapPix12=180
        self.overlapPix23=160
        self.overlapPix34=200
        self.overlapPix45=200 # Guess, needs to be tuned
        
        self.smoothingPix=1# Number of pixels to smooth over
        # Create gradient arrays for Left and Right
        self.gradientArrLeft=np.linspace(1.0,0.0, self.smoothingPix, endpoint=True)
        self.gradientArrLeft=np.tile(self.gradientArrLeft,(512,1))
        self.gradientArrRight=np.linspace(0.0,1.0, self.smoothingPix, endpoint=True)
        self.gradientArrRight=np.tile(self.gradientArrRight,(512,1))

        self.homographyMat1_2=np.array([[1,0,0],[0,1,-1],[0.00024,0.00,0.853]])
        self.homographyMat2_3=np.array([[1,0,0],[0,1,-5],[0.00012,0.00,0.83]])
        self.homographyMat3_4=np.array([[1,0,0],[0,1,0],[-0.00036,0.00,1]])
        self.homographyMat4_5=np.array([[1,0,0],[0,1,0],[-0.00036,0.00,1]]) # Guess, needs to be tuned

# Subscribe to AGC Gains of all cameras
        rospy.Subscriber('/flir_boson1/MinAgcTopic', std_msgs.msg.UInt16, self.MinAgc1)
        rospy.Subscriber('/flir_boson2/MinAgcTopic', std_msgs.msg.UInt16, self.MinAgc2)
        rospy.Subscriber('/flir_boson3/MinAgcTopic', std_msgs.msg.UInt16, self.MinAgc3)
        rospy.Subscriber('/flir_boson4/MinAgcTopic', std_msgs.msg.UInt16, self.MinAgc4)
        rospy.Subscriber('/flir_boson5/MinAgcTopic', std_msgs.msg.UInt16, self.MinAgc5)
        rospy.Subscriber('/flir_boson1/MaxAgcTopic', std_msgs.msg.UInt16, self.MaxAgc1)
        rospy.Subscriber('/flir_boson2/MaxAgcTopic', std_msgs.msg.UInt16, self.MaxAgc2)
        rospy.Subscriber('/flir_boson3/MaxAgcTopic', std_msgs.msg.UInt16, self.MaxAgc3)
        rospy.Subscriber('/flir_boson4/MaxAgcTopic', std_msgs.msg.UInt16, self.MaxAgc4)
        rospy.Subscriber('/flir_boson5/MaxAgcTopic', std_msgs.msg.UInt16, self.MaxAgc5) 

 # Build Images, correct for AGC (only works with TIFF mode)             
        rospy.Subscriber('/flir_boson1/image_rect', Image, self.buildimage1)
        rospy.Subscriber('/flir_boson2/image_rect', Image, self.buildimage2)
        rospy.Subscriber('/flir_boson3/image_rect', Image, self.buildimage3)
        rospy.Subscriber('/flir_boson4/image_rect', Image, self.buildimage4)
        rospy.Subscriber('/flir_boson5/image_rect', Image, self.buildimage5)


  # Build Images and correct for gain:      
    def buildimage1(self,data):
        self.image1=self.bridge.imgmsg_to_cv2(data, "mono8")
        self.image1=np.multiply(self.image1,((self.MaxAgcVal1-self.MinAgcVal1)/255)*np.ones(self.image1.shape))+255
        #self.image1= cv2.normalize(src=self.image1, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) 
    def buildimage2(self,data):
        self.image2=self.bridge.imgmsg_to_cv2(data, "mono8")
        self.image2=np.multiply(self.image2,((self.MaxAgcVal2-self.MinAgcVal2)/255)*np.ones(self.image2.shape))+255
    
    def buildimage3(self,data):
        self.image3=self.bridge.imgmsg_to_cv2(data, "mono8")
        self.image3=np.multiply(self.image3,((self.MaxAgcVal3-self.MinAgcVal3)/255)*np.ones(self.image3.shape))+255

    def buildimage4(self,data):
        self.image4=self.bridge.imgmsg_to_cv2(data, "mono8")
        self.image4=np.multiply(self.image4,((self.MaxAgcVal4-self.MinAgcVal4)/255)*np.ones(self.image4.shape))+255

        
    def buildimage5(self,data):
        self.image5=self.bridge.imgmsg_to_cv2(data, "mono8")
        self.image5=np.multiply(self.image5,((self.MaxAgcVal5-self.MinAgcVal5)/255)*np.ones(self.image5.shape))+255
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
        #Normalize the Image
        self.Panorama= cv2.normalize(src=self.Panorama, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) 
        # Conversion to RGB
        self.Panorama=cv2.cvtColor(self.Panorama,cv2.COLOR_GRAY2RGB)    
        # Publishers
        self.PanoPub.publish(self.bridge.cv2_to_imgmsg(self.Panorama, "rgb8"))
        #self.TempWarpPub.publish(self.bridge.cv2_to_imgmsg(self.Warped1_2, "mono8")) # For Debugging
        rospy.loginfo('Published Panorama')
        
        
         


if __name__=='__main__':
    main()


