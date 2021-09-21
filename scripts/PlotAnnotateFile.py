#!/usr/bin/env python
# Uses annotation Text file to plot onto Thermal Panorama, to check quality of annotations
from numpy.core.fromnumeric import shape
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
import math
from sensor_msgs.msg import Image
from radar_msgs.msg import RadarTrackArray
from derived_object_msgs.msg import ObjectWithCovarianceArray
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2



def main():
    rospy.init_node('gTruthToFile', anonymous=True)
    vsInst=gTruth()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

class gTruth:

    def __init__(self):
        self.bridge=CvBridge()
        self.frameId=1
        self.lineId=0
        self.font=cv2.FONT_HERSHEY_SIMPLEX 
        with open('plotSeq.txt') as f:
            self.AllLines = f.readlines()
        # print(len(self.AllLines))
        self.image_pub = rospy.Publisher("groundTruthImage",Image,queue_size=1)
        # Caliberation: Roughly measured in car.
        rospy.Subscriber('/Thermal_Panorama', Image, self.image)
        rospy.Subscriber("/os_cloud_node/points",PointCloud2, self.readFile)
        
        # filePathPrefix=str("/home/vamsi/Tracking/py-motmetrics/motmetrics/data/seq1/gt/")


    def readFile(self,data):
        # Read the next frame and get all tracks in that frame
        
        
        lineData=self.AllLines[self.lineId].split()
        # print(int(lineData[0]))
        # print(self.frameId)
        # print(int(lineData[0])==self.frameId)

        while int(lineData[0])==self.frameId:
            self.plotter(lineData)
            self.lineId=self.lineId+1
            lineData=self.AllLines[self.lineId].split()
        else:
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.BBImage, "rgb8"))
                # self.lineId=self.lineId+1
                self.frameId=self.frameId+1
            except:
                pass

    def image(self,data):
        
        self.RawImage=self.bridge.imgmsg_to_cv2(data, "rgb8")
        self.ImageExists=1


    def plotter(self,lineData):
        self.BBImage = self.RawImage
       # imageTemp=cv2.cvtColor(imageTemp,cv2.COLOR_GRAY2RGB)

        trackId=int(lineData[1])
        bb_left=int(lineData[2])
        bb_top=int(lineData[3])
        # print(trackId)
        bb_width=int(lineData[4])
        bb_height=int(lineData[5])
        pt1=((bb_left-bb_width/2),(bb_top-bb_height/2))
        pt2=((bb_left+bb_width/2),(bb_top+bb_height/2))
        color = np.array([0, 255, 0])
        thickness = 2

        self.BBImage=cv2.rectangle(self.BBImage,pt1,pt2,color.tolist(), thickness)
        self.BBImage=cv2.putText(self.BBImage,str(trackId),(bb_left,bb_top),self.font,1,(255,105,180),2)
        # print(self.BBImage.shape)
        

        

      
        


if __name__=='__main__':

	main()
