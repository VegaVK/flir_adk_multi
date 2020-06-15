#!/usr/bin/env python

import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
import math
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
        self.image_pub = rospy.Publisher("radar_image",Image,queue_size=100)
        self.image1=np.array([1])
        # Caliberation: Roughly measured in car.
        self.delta_x = 2.3114
        self.delta_y = 0 # Assuming that the radar and camera are on same centerline
        self.delta_z = 1.0414/2
        self.H_FOV=50
        self.V_FOV=41 #Calculated based on aspect ratio
        self.HorzOffset=-30 # Manual horizontal (Y-direction) offset for radar in pixels
        self.VertOffset=-10 # Manual vertical (Z-direction) offset for radar in pixels
        #-self.RadarTracks[idx]['RadarX']*np.sin(np.radians(4)) 
        rospy.loginfo('Published fused image')
        # rospy.Subscriber('vehicle/steering_report', dbw_mkz_msgs.msg.SteeringReport, self.VehSpeedCallback)
        # rospy.Subscriber("/as_tx/objects",ObjectWithCovarianceArray, self.radarObj)
        rospy.Subscriber('/flir_boson3/image_rect', Image, self.image)
        rospy.Subscriber("/as_tx/radar_tracks/",RadarTrackArray, self.radarTracks)
        


    # x in Radar data is along longitudinal direction and y is lateral (right to left?), with zero at center of car
    # z is just dummy variable in radar data.  
      
    # def VehSpeedCallback(self,data):
        # self.VehSpeed=data.speed
        # print(data.speed)

    def radarTracks(self,data):
        n = len(data.tracks)

        self.RadarTracks=np.zeros((n,1),dtype=[('id', np.uint16),('RadarX', np.float64),('RadarY', np.float64),('RadarZ', np.float64),('RadarVelX', np.float64),('RadarVelY', np.float64),('RadarAccelX', np.float64),('RadarAccelY', np.float64)])
        for idx in range(n):
            self.RadarTracks[idx]['id']= data.tracks[idx].track_id
            self.RadarTracks[idx]['RadarX'] = data.tracks[idx].track_shape.points[1].x+self.delta_x
            self.RadarTracks[idx]['RadarY'] = (data.tracks[idx].track_shape.points[1].y+data.tracks[idx].track_shape.points[2].y)/2
            self.RadarTracks[idx]['RadarZ'] = data.tracks[idx].track_shape.points[1].z-self.delta_z
            self.RadarTracks[idx]['RadarVelX'] = data.tracks[idx].linear_velocity.x
            self.RadarTracks[idx]['RadarVelY'] = data.tracks[idx].linear_velocity.y
            self.RadarTracks[idx]['RadarAccelX']=data.tracks[idx].linear_acceleration.x
            self.RadarTracks[idx]['RadarAccelY']=data.tracks[idx].linear_acceleration.y
        DelIndexArray=[]
        for jdx in range(n):
            # print(jdx)
            if abs(self.RadarTracks[idx]['RadarAccelX'])<=0.05:
                DelIndexArray.append(int(jdx))
        # IF we want to delete:
        # self.RadarTracks=np.delete(self.RadarTracks,DelIndexArray)



                
    def image(self,data):
        
        self.RawImage=self.bridge.imgmsg_to_cv2(data, "mono8")
        
        self.fusefun() 
        

    def fusefun(self):
        imageTemp = self.RawImage
        n=len(self.RadarTracks)
        self.RadarAnglesH=np.zeros((n,1))
        self.RadarAnglesV=np.zeros((n,1))
        # Camera Coordinates: X is horizontal, Y is vertical starting from left top corner
        for idx in range(n):
            self.RadarAnglesH[idx]=-np.degrees(np.arctan(np.divide(self.RadarTracks[idx]['RadarY'],self.RadarTracks[idx]['RadarX'])))
            self.RadarAnglesV[idx]=np.abs(np.degrees(np.arctan(np.divide(self.RadarTracks[idx]['RadarZ'],self.RadarTracks[idx]['RadarX'])))) #will always be negative, so correct for it
        self.CameraX=self.RadarAnglesH*(self.RawImage.shape[1]/self.H_FOV) + self.RawImage.shape[1]/2 +self.HorzOffset# Number of pixels per degree,adjusted for shifting origin from centerline to top left
        self.CameraY=self.RadarAnglesV*(self.RawImage.shape[0]/self.V_FOV) +256 +self.VertOffset -self.RadarTracks[idx]['RadarX']*np.sin(np.radians(4)) # Number of pixels per degree,adjusted for shifting origin from centerline to top left
        

        for idx in range(len(self.RadarAnglesH)):
            if (self.CameraX[idx]<=self.RawImage.shape[1]):
                cv2.circle(imageTemp, (int(self.CameraX[idx]),int(self.CameraY[idx])), 3, (255,105,180))

        
        imageTemp=cv2.cvtColor(imageTemp,cv2.COLOR_GRAY2RGB)
        for idx in range(len(self.RadarAnglesH)):
            if (self.CameraX[idx]<=self.RawImage.shape[1]):
                cv2.circle(imageTemp, (int(self.CameraX[idx]),int(self.CameraY[idx])), 10, (255,105,180),3)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(imageTemp, "rgb8"))

      
        


if __name__=='__main__':

	main()
