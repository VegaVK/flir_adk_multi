#!/usr/bin/env python

import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
from math import *
from sensor_msgs.msg import Image
from delphi_esr_msgs.msg import EsrEthTx
from radar_msgs.msg import RadarTrackArray
from derived_object_msgs.msg import ObjectWithCovarianceArray
from std_msgs.msg import String



def main():
    rospy.init_node('fusion', anonymous=True)
    vsInst=radar_img()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

class radar_img:

    def __init__(self):
        self.bridge=CvBridge()
        self.image_pub = rospy.Publisher("fused_image",Image,queue_size=100)
        self.image1=np.array([1])
        
       
        

        rospy.loginfo('Published fused image')
        rospy.Subscriber('/flir_boson3/image_rect', Image, self.image)
        # rospy.Subscriber("/parsed_tx/esr_eth_tx_msg",EsrEthTx, self.radar)
        rospy.Subscriber("/as_tx/objects",ObjectWithCovarianceArray, self.radar)


        
    def radar(self,data):

        n = len(data.objects)
        para_y = 0.14
        para_phi = 5

        self.id = np.empty([n])
        self.x = np.empty([n])
        self.y = np.empty([n])
        self.thi = np.empty([n])
        self.phi = np.empty([n])
        # print("haha")



        for i in range(len(data.objects)):
            self.id[i] = data.objects[i].id
            self.x[i] = data.objects[i].pose.pose.position.x
            self.y[i] = data.objects[i].pose.pose.position.y
            self.thi[i] = 180/3.14*(atan(self.y[i]/self.x[i]))
            self.phi[i] = 180/3.14*(atan(para_y/self.x[i])) + para_phi
            # print(180/3.14*(atan(para_y/self.x[i]) + para_phi))
            # if data.objects[i].object_classified == True:
            #     print("True")


     
        

    def image(self,data):
        
        self.image1=self.bridge.imgmsg_to_cv2(data, "mono8")
        self.fusefun() 
        

    def fusefun(self):
    	image2 = self.image1

        # print(len(self.thi))

    	for i in range(len(self.thi)):
            if int(320-12.8*self.thi[i]) > 0 and int(320-12.8*self.thi[i]) < 640:
                if int(256 + 512/40*self.phi[i]) < 512:
                    cen = (int(320-12.8*self.thi[i]),int(256 + 512/40*self.phi[i]))
                    # print(int(256 + 512/40*self.phi[i]))
                    cv2.circle(image2, cen, 3, 255)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(image2, "mono8"))
        #print(self.Final.shape)
      
        


if __name__=='__main__':

	main()
    
#    rate = rospy.Rate(10)
#    hello_str = "hello world"
 #   rospy.loginfo(hello_str)
 #   pub.publish(hello_str)
 #   rospy.spin()
 #   exit(0) 