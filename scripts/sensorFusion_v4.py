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
from darknet_ros_msgs.msg import BoundingBoxes


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
        self.boxes = 0
        self.thi = np.array([1])
        self.phi = np.array([1])
        
       
        

        rospy.loginfo('Published fused image')
        
        rospy.Subscriber("/as_tx/objects",ObjectWithCovarianceArray, self.radar)
        rospy.Subscriber("/darknet_ros/detection_image",Image,self.image)
        rospy.Subscriber("/darknet_ros/bounding_boxes",BoundingBoxes,self.bounding_boxes)


    def bounding_boxes(self,data):
        self.boxes = len(data.bounding_boxes)
        self.start = []
        self.end = []
        self.center = []

        for i in range(self.boxes):
            self.start.append((data.bounding_boxes[i].xmin,data.bounding_boxes[i].ymin))
            self.end.append((data.bounding_boxes[i].xmax,data.bounding_boxes[i].ymax)) 
            self.center.append(((0.5*(data.bounding_boxes[i].xmin + data.bounding_boxes[i].xmax)),(0.5*(data.bounding_boxes[i].ymin+ data.bounding_boxes[i].ymax))))




    def radar(self,data):

        n = len(data.objects)
        para_y = 1              #verticle distance between camera and radar
        para_phi = 4            #pitch offset of camera from the forward horizontal axis in deg (estimate)

        self.id = np.empty([n])
        self.x = np.empty([n])
        self.y = np.empty([n])
        self.thi = np.empty([n])
        self.phi = np.empty([n])
        self.points = []

        for i in range(len(data.objects)):
            self.id[i] = data.objects[i].id
            self.x[i] = 1.5*data.objects[i].pose.pose.position.x
            self.y[i] = data.objects[i].pose.pose.position.y
            self.thi[i] = 180/3.14*(atan(self.y[i]/self.x[i]))
            self.phi[i] = 180/3.14*(atan(para_y/self.x[i])) + para_phi

            if int(320-12.8*self.thi[i]) > 0 and int(320-12.8*self.thi[i]) < 640 and int(256 + 512/40*self.phi[i]) < 512:
                self.points.append((int(320-12.8*self.thi[i]),int(256 + 512/40*self.phi[i])))   
     
        

    def image(self,data):
        
        image=self.bridge.imgmsg_to_cv2(data, "rgb8")
        
        for i in range(len(self.points)):
            cv2.circle(image, self.points[i], 3, (255,0,0))

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "rgb8"))

        


if __name__=='__main__':

	main()
