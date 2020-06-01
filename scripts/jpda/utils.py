#!/usr/bin/env python
# File for custom messages etc

import rospy
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from darknet_ros_msgs.msg import BoundingBoxes



class CamObj:
    def __init__(self):
        self.stamp= Header().stamp # Gives time in nanoseconds
        self.PxWidth=[]
        self.PxHeight=[]
        self.Position=Pose().position # To be calculated later, if req
        self.id=[] #ID of object, person=0, bicycle=1, all other veh =2
class RadarObj:
    def __init__(self):
        self.stamp=Header().stamp # Gives time in nanoseconds
        self.vx=[] #Relative Velocity, so have to ADD vehicle velocity.
        self.vy=[]
        self.pose=Pose() # Also relative distances from ego frame


    