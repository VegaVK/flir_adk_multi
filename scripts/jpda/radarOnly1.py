#!/usr/bin/env python
import rospy
import numpy as np
import sys
import os
# import dbw_mkz_msgs.msg
# from sensor_msgs.msg import Image
# from delphi_esr_msgs.msg import EsrEthTx
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from darknet_ros_msgs.msg import BoundingBoxes
from nuscenes2bag.msg import RadarObjects
sys.path.append(".")
from utils import RadarObj
from utils import CamObj
from flir_adk_multi.msg import trackArray
def main():
    rospy.init_node('jpda', anonymous=True)
    fusInst=jpda_class()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
class jpda_class:

    def __init__(self):
        self.Predictions=rospy.Publisher("JPDA",trackArray, queue_size=100) 
        self.YoloClassList=[0,1,2,3,5,7]
        rospy.Subscriber('darknet_ros/bounding_boxes', BoundingBoxes, self.CamMsrmts)
        rospy.Subscriber('/radar_front', RadarObjects, self.RdrMsrmtsNuSc)
        # rospy.Subscriber('/radar_front', BoundingBoxes, self.RdrMsrmtsMKZ)
        
        # trackManager()
        # nearestNAlg()
        # JPDAalg()



    def CamMsrmts(self,data):
        self.CamReadings=[]
        for idx in range(len(data.bounding_boxes)):
            #print(data.bounding_boxes[2].id)
            if data.bounding_boxes[idx].id in self.YoloClassList:
                self.CamReadings.append(CamObj())
                self.CamReadings[-1].stamp=data.header.stamp
                self.CamReadings[-1].PxWidth=(data.bounding_boxes[idx].xmax-data.bounding_boxes[idx].xmin)
                self.CamReadings[-1].PxHeight=(data.bounding_boxes[idx].ymax-data.bounding_boxes[idx].ymin)
                self.CamReadings[-1].id=data.bounding_boxes[idx].id
        # for elem in self.TempReadings:
        #     if elem:
        #         self.CamReadings.append(elem)
        #print(len(self.CamReadings))

    #TODO: create new RdrMsrmtsMKZ to work with MKZ.
    #def RdrMsrmtsNuSc(self,data): 


    def RdrMsrmtsNuSc(self,data): 
        self.RdrReadings=[]
        for idx in range(len(data.objects)):
            self.RdrReadings.append(RadarObj())
            self.RdrReadings[-1].pose=data.objects[idx].pose
            self.RdrReadings[-1].vx=data.objects[idx].vx
            self.RdrReadings[-1].vy=data.objects[idx].vy
            self.RdrReadings[-1].stamp=data.header.stamp
        print(self.RdrReadings[3].stamp)


if __name__=='__main__':

	main()
    

            
        
