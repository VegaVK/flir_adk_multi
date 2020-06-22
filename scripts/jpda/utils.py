#!/usr/bin/env python
# File for custom messages etc

import rospy
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from darknet_ros_msgs.msg import BoundingBoxes
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension


def Mat_buildROS(matrixDat): # Converts numpy array to ROS Array
        temp=Float32MultiArray()
        MatSize=matrixDat.shape[0]*matrixDat.shape[1]
        temp.data=matrixDat.reshape([MatSize])
        temp.layout.data_offset=0
        temp.layout.dim=[MultiArrayDimension(), MultiArrayDimension()]
        temp.layout.dim[0].label="Dim0"
        temp.layout.dim[1].label="Dim1"
        temp.layout.dim[0].size=matrixDat.shape[0]
        temp.layout.dim[1].size=matrixDat.shape[1]
        temp.layout.dim[0].stride=MatSize
        temp.layout.dim[1].stride=matrixDat.shape[1]
        return temp

def Mat_extractROS(MultiArrMsg): # Extracts numpy array from ROS MultiArray type  (for 2D Arrays)
    temp=np.array(MultiArrMsg.data)
    temp=temp.reshape((MultiArrMsg.layout.dim[0].size,MultiArrMsg.layout.dim[1].size))
    return temp
        

class CamObj:
    def __init__(self):
        self.stamp= Header().stamp # Gives time in nanoseconds
        self.PxWidth=[]
        self.PxHeight=[]
        self.XPix=[] # Location of the center of the box
        self.YPix=[]
        # self.Area=[] 
        # self.Ratio=[]
        # self.Position=Pose().position # To be calculated later, if req
        self.id=[] #ID of object, person=0, bicycle=1, all other veh =2
        self.confidence=[] # Probability of detection
class RadarObj:
    def __init__(self):
        self.header=Header() # Gives time in nanoseconds
        self.vx=[] #Relative Velocity, so have to ADD vehicle velocity.
        self.vy=[]
        self.vx_comp=[]
        self.vy_comp=[]
        self.pose=Pose() # Also relative distances from ego frame
class RadarObjMKZ:
    def __init__(self):
        self.id=[]
        self.header=Header() # Gives time in nanoseconds
        self.vx=[] #Relative Velocity, so have to ADD vehicle velocity.
        self.vy=[]
        self.vx_comp=[]
        self.vy_comp=[]
        self.pose=Pose() # Also relative distances from ego frame


    