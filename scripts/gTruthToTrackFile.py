#!/usr/bin/env python

import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
import math
from sensor_msgs.msg import Image
from radar_msgs.msg import RadarTrackArray
from derived_object_msgs.msg import ObjectWithCovarianceArray
from std_msgs.msg import String



def main():
    rospy.init_node('gTruthToFile', anonymous=True)
    vsInst=gTruth()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        self.DestF.close()
        print("Shutting down")

class gTruth:

    def __init__(self):
        self.bridge=CvBridge()
        self.image_pub = rospy.Publisher("radar_image",Image,queue_size=100)
        # Caliberation: Roughly measured in car.
        rospy.Subscriber("/as_tx/cam_ground_truth_objects/",ObjectWithCovarianceArray, self.createTracks)
        rospy.Subscriber('/Thermal_Panorama', Image, self.image)
        filePathPrefix=str("/home/vamsi/Tracking/py-motmetrics/motmetrics/data/seq1/gt/")
        self.delta_x = 0
        self.delta_y = 0 # Assuming that the radar and camera are on same centerline
        self.delta_z = 1.0414/2
        self.H_FOV=190
        self.V_FOV=41 #Calculated based on aspect ratio
        self.HorzOffset=0 # Manual horizontal (Y-direction) offset for radar in pixels
        self.VertOffset=-30 # Manual vertical (Z-direction) offset for radar in pixels
        self.DestF=open((filePathPrefix+'gt'+'.txt'),"w")
        self.ImageExists=0
        self.BBheight=90
        self.BBWidth=90 # For now, static
        self.FrameInit=1
        self.tracks=ObjectWithCovarianceArray()
        #-self.RadarTracks[idx]['RadarX']*np.sin(np.radians(4)) 
        


    # x in Radar data is along longitudinal direction and y is lateral (right to left?), with zero at center of car
    # z is just dummy variable in radar data.  
      
    # def VehSpeedCallback(self,data):
        # self.VehSpeed=data.speed
        # print(data.speed)
    def createTracks(self,dataIN):
        self.tracks=dataIN

    def writeToFile(self):
        data=self.tracks
        # self.Readoings=[]
        # n=len(self.RadarTracks)
        frame=self.FrameInit
        self.FrameInit+=1
        print(frame)
        RadarAnglesH=0.0
        RadarAnglesV=0.0
        for idx in range(len(data.objects)):
            if (data.objects[idx].pose.pose.position.x==0.0) and (data.objects[idx].pose.pose.position.y==0.0) and (data.objects[idx].pose.covariance[0]==0.0):
                continue #Zero entry, so skip it
            else: #write to file
                # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                
                id=int(data.objects[idx].id)
                RadarX=data.objects[idx].pose.pose.position.x+self.delta_x
                RadarY=data.objects[idx].pose.pose.position.y
                RadarZ=0.0+self.delta_z
                RadarAnglesH=-np.degrees(np.arctan(np.divide(RadarY,RadarX)))
                RadarAnglesV=np.abs(np.degrees(np.arctan(np.divide(RadarZ,RadarX)))) #will always be negative, so correct for it
                

                
                if self.ImageExists==1:
                    imageTemp = self.RawImage
                    # print(imageTemp.shape)
                    CameraX=RadarAnglesH*(self.RawImage.shape[1]/self.H_FOV) + self.RawImage.shape[1]/2 +self.HorzOffset# Number of pixels per degree,adjusted for shifting origin from centerline to top left
                    CameraY=RadarAnglesV*(self.RawImage.shape[0]/self.V_FOV) +256 +self.VertOffset -RadarX*np.sin(np.radians(4)) # Number of pixels per degree,adjusted for shifting origin from centerline to top left
    
                    # imageTemp=cv2.cvtColor(imageTemp,cv2.COLOR_GRAY2RGB)
                   
                    if (CameraX<=self.RawImage.shape[1]):
                        start=(int(CameraX-self.BBWidth/2), int(CameraY-self.BBheight/2))
                        end= (int(CameraX+self.BBWidth/2), int(CameraY+self.BBheight/2))
                        imageTemp=cv2.rectangle(imageTemp,start,end,(0,0,255),2)
                        cv2.circle(imageTemp, (int(CameraX),int(CameraY)), 10, (255,255,102),3)
                        self.image_pub.publish(self.bridge.cv2_to_imgmsg(imageTemp, "rgb8"))
                    
                    #Write to File
                    bb_left=int(CameraX)
                    bb_top=int(CameraY)
                    bb_width=int(self.BBWidth)
                    bb_height=int(self.BBheight)
                    x=-1 # Fillers
                    y=-1
                    z=-1
                    conf=1
                    outLine=str(frame)+' '+str(id)+' '+str(bb_left)+' '+str(bb_top)+' '+str(bb_width)+' '+str(bb_height)+' '+str(conf)+' '+str(x)+' '+str(y)+' '+str(z)+'\n'
                    
                    self.DestF.write(outLine)
                    

                


                
        #     self.RdrReadings=np.append(self.RdrReadings,RadarObjMKZ())
        #     self.RdrReadings[-1].pose=data.objects[idx].pose.pose
        #     self.RdrReadings[-1].vx=data.objects[idx].twist.twist.linear.x
        #     self.RdrReadings[-1].vy=data.objects[idx].twist.twist.linear.y
        #     self.RdrReadings[-1].vx_comp=self.velX+data.objects[idx].twist.twist.linear.x
        #     self.RdrReadings[-1].vy_comp=self.velY+data.objects[idx].twist.twist.linear.y
        #     self.RdrReadings[-1].header=data.objects[idx].header
        #     self.RdrReadings[-1].id=data.objects[idx].id
            
        # self.RdrReadings=np.asarray(self.RdrReadings)





        # self.RadarTracks=np.zeros((n,1),dtype=[('id', np.uint16),('RadarX', np.float64),('RadarY', np.float64),('RadarZ', np.float64),('RadarVelX', np.float64),('RadarVelY', np.float64),('RadarAccelX', np.float64),('RadarAccelY', np.float64)])
        # for idx in range(n):
        #     self.RadarTracks[idx]['id']= data.tracks[idx].track_id
        #     self.RadarTracks[idx]['RadarX'] = data.tracks[idx].track_shape.points[1].x+self.delta_x
        #     self.RadarTracks[idx]['RadarY'] = (data.tracks[idx].track_shape.points[1].y+data.tracks[idx].track_shape.points[2].y)/2
        #     self.RadarTracks[idx]['RadarZ'] = data.tracks[idx].track_shape.points[1].z-self.delta_z
        #     self.RadarTracks[idx]['RadarVelX'] = data.tracks[idx].linear_velocity.x
        #     self.RadarTracks[idx]['RadarVelY'] = data.tracks[idx].linear_velocity.y
        #     self.RadarTracks[idx]['RadarAccelX']=data.tracks[idx].linear_acceleration.x
        #     self.RadarTracks[idx]['RadarAccelY']=data.tracks[idx].linear_acceleration.y
        # DelIndexArray=[]
        # for jdx in range(n):
        #     # print(jdx)
        #     if abs(self.RadarTracks[idx]['RadarAccelX'])<=0.05:
        #         DelIndexArray.append(int(jdx))
        # # IF we want to delete:
        # # self.RadarTracks=np.delete(self.RadarTracks,DelIndexArray)



                
    def image(self,data):
        
        self.RawImage=self.bridge.imgmsg_to_cv2(data, "rgb8")
        self.ImageExists=1
        self.writeToFile()
        
        # self.plotter() 
        

    # def plotter(self):
    #     imageTemp = self.RawImage
    #     n=len(self.RadarTracks)
    #     self.RadarAnglesH=np.zeros((n,1))
    #     self.RadarAnglesV=np.zeros((n,1))
    #     self.MovingObjStatus=np.zeros((n,1)) # To check if object is moving or not
    #     # Camera Coordinates: X is horizontal, Y is vertical starting from left top corner
    #     for idx in range(len(self.RadarTracks)):
    #         if np.sqrt((self.RadarTracks[idx]['RadarAccelX'])**2 + (self.RadarTracks[idx]['RadarAccelY'])**2)>=0.05:
    #             self.MovingObjStatus[idx]=1
    #         else:
    #             self.MovingObjStatus[idx]=0
    #         self.RadarAnglesH[idx]=-np.degrees(np.arctan(np.divide(self.RadarTracks[idx]['RadarY'],self.RadarTracks[idx]['RadarX'])))
    #         self.RadarAnglesV[idx]=np.abs(np.degrees(np.arctan(np.divide(self.RadarTracks[idx]['RadarZ'],self.RadarTracks[idx]['RadarX'])))) #will always be negative, so correct for it
    #     self.CameraX=self.RadarAnglesH*(self.RawImage.shape[1]/self.H_FOV) + self.RawImage.shape[1]/2 +self.HorzOffset# Number of pixels per degree,adjusted for shifting origin from centerline to top left
    #     self.CameraY=self.RadarAnglesV*(self.RawImage.shape[0]/self.V_FOV) +256 +self.VertOffset -self.RadarTracks[idx]['RadarX']*np.sin(np.radians(4)) # Number of pixels per degree,adjusted for shifting origin from centerline to top left
        


        
    #     imageTemp=cv2.cvtColor(imageTemp,cv2.COLOR_GRAY2RGB)
    #     for idx in range(len(self.RadarAnglesH)):
    #         if (self.CameraX[idx]<=self.RawImage.shape[1]):
    #             if self.MovingObjStatus[idx]==1:
    #                 cv2.circle(imageTemp, (int(self.CameraX[idx]),int(self.CameraY[idx])), 10, (255,105,180),3)
    #             # print(str(self.RadarTracks[idx]['id'][0]))
    #                 cv2.putText(imageTemp,str(self.RadarTracks[idx]['id'][0]),(self.CameraX[idx],self.CameraY[idx]),self.font,1,(255,105,180),2)
    #             else:
    #                 cv2.circle(imageTemp, (int(self.CameraX[idx]),int(self.CameraY[idx])), 10, (255,255,102),3)

    #     self.image_pub.publish(self.bridge.cv2_to_imgmsg(imageTemp, "rgb8"))

      
        


if __name__=='__main__':

	main()
