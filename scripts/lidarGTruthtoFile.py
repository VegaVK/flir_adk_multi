#!/usr/bin/env python
# Uses lidar annotations from provided .yaml file to convert to fram-wise text file for MOT

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
    rospy.init_node('lidarGTruthToFile', anonymous=True)
    vsInst=gTruth()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        self.DestF.close()
        print("Shutting down")

class gTruth:

    def __init__(self):
        self.bridge=CvBridge()
        self.image_pub = rospy.Publisher("groundTruthImage",Image,queue_size=100)
        # Caliberation: Roughly measured in car.
        rospy.Subscriber("/as_tx/cam_ground_truth_objects/",ObjectWithCovarianceArray, self.createTracks)
        rospy.Subscriber('/Thermal_Panorama', Image, self.image)
        filePathPrefix=str("/home/vamsi/Tracking/py-motmetrics/motmetrics/data/seq1/gt/")
        self.delta_x = 0
        self.delta_y = 0 # Assuming that the radar and camera are on same centerline
        self.delta_z = 1.1 # From Ego vehicle to Camera Frame
        self.H_FOV=190
        self.V_FOV=41 #Calculated based on aspect ratio
        self.HorzOffsetGain=2.5 # Manual horizontal (Y-direction) offset GAIN (changes with position) for radar in pixels
        self.BBWidthGain = 2.5 # Heuristic gain  to adjust bounding box width
        self.VertOffset=0 # Manual vertical (Z-direction) offset for radar in pixels
        self.DestF=open((filePathPrefix+'gtLidar'+'.txt'),"w")
        self.ImageExists=0
        self.CarMeshW=2.2 # Assume same for all cars
        self.CarMeshH=1.8
        # self.BBheight=90
        # self.BBWidth=90 # For now, static
        self.FrameInit=1
        self.tracks=ObjectWithCovarianceArray()
        #-self.RadarTracks[idx]['EgoMeasX']*np.sin(np.radians(4)) 
        


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
        # print(frame)
        CameraAnglesH=0.0
        CameraAnglesV=0.0
        for idx in range(len(data.objects)):
            
            if (data.objects[idx].pose.pose.position.x==0.0) and (data.objects[idx].pose.pose.position.y==0.0) and (data.objects[idx].pose.covariance[0]==0.0):
                continue #Zero entry, so skip it
            else: #write to file
                print(idx)
                # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                
                id=int(data.objects[idx].id)
                EgoMeasX=data.objects[idx].pose.pose.position.x+self.delta_x
                EgoMeasY=data.objects[idx].pose.pose.position.y
                EgoMeasZ=0.0-self.delta_z/2
                CameraAnglesH=-np.degrees(np.arctan(np.divide(EgoMeasY,EgoMeasX))) #will always be negative, so correct for it
                CameraAnglesV=-np.abs(np.degrees(np.arctan(np.divide(EgoMeasZ,EgoMeasX))))
                

                
                if self.ImageExists==1:
                    imageTemp = self.RawImage
                    # print(imageTemp.shape)
                    CameraX=CameraAnglesH*(self.RawImage.shape[1]/self.H_FOV) + self.RawImage.shape[1]/2 +self.HorzOffsetGain*CameraAnglesH# Number of pixels per degree,adjusted for shifting origin from centerline to top left
                    # CameraY=CameraAnglesV*(self.RawImage.shape[0]/self.V_FOV) +256 +self.VertOffset -EgoMeasX*np.sin(np.radians(4)) # Number of pixels per degree,adjusted for shifting origin from centerline to top left
                    CameraY=CameraAnglesV*(self.RawImage.shape[0]/self.V_FOV) +256 +self.VertOffset
                    BBHeight=-np.degrees(np.arctan(np.divide(self.CarMeshH,EgoMeasX)))*(self.RawImage.shape[0]/self.V_FOV) 
                    BBWidth=np.degrees(np.arctan(np.divide(self.CarMeshW/2.0,EgoMeasX)))*2.0*(self.RawImage.shape[0]/self.V_FOV) +abs(self.BBWidthGain*CameraAnglesH)# Since heading info is not available, assume =0 and use heuristic
                    # imageTemp=cv2.cvtColor(imageTemp,cv2.COLOR_GRAY2RGB)
                   
                    if (CameraX<=self.RawImage.shape[1]):
                        start=(int(CameraX-BBWidth/2), int(CameraY-BBHeight/2))
                        end= (int(CameraX+BBWidth/2), int(CameraY+BBHeight/2))
                        imageTemp=cv2.rectangle(imageTemp,start,end,(0,0,255),2)
                        # cv2.circle(imageTemp, (int(CameraX),int(CameraY)), 10, (255,255,102),3)
                        self.image_pub.publish(self.bridge.cv2_to_imgmsg(imageTemp, "rgb8"))
                    
                    #Write to File
                    bb_left=int(CameraX)
                    bb_top=int(CameraY)
                    bb_width=int(BBWidth)
                    bb_height=int(BBHeight)
                    x=-1 # Fillers
                    y=-1
                    z=-1
                    conf=1
                    outLine=str(frame)+' '+str(id)+' '+str(bb_left)+' '+str(bb_top)+' '+str(bb_width)+' '+str(bb_height)+' '+str(conf)+' '+str(x)+' '+str(y)+' '+str(z)+'\n'
                    
                    self.DestF.write(outLine)
                    

                



                
    def image(self,data):
        
        self.RawImage=self.bridge.imgmsg_to_cv2(data, "rgb8")
        self.ImageExists=1
        self.writeToFile()
        
        # self.plotter() 
        

    # def plotter(self):
    #     imageTemp = self.RawImage
    #     n=len(self.RadarTracks)
    #     self.CameraAnglesH=np.zeros((n,1))
    #     self.CameraAnglesV=np.zeros((n,1))
    #     self.MovingObjStatus=np.zeros((n,1)) # To check if object is moving or not
    #     # Camera Coordinates: X is horizontal, Y is vertical starting from left top corner
    #     for idx in range(len(self.RadarTracks)):
    #         if np.sqrt((self.RadarTracks[idx]['RadarAccelX'])**2 + (self.RadarTracks[idx]['RadarAccelY'])**2)>=0.05:
    #             self.MovingObjStatus[idx]=1
    #         else:
    #             self.MovingObjStatus[idx]=0
    #         self.CameraAnglesH[idx]=-np.degrees(np.arctan(np.divide(self.RadarTracks[idx]['EgoMeasY'],self.RadarTracks[idx]['EgoMeasX'])))
    #         self.CameraAnglesV[idx]=np.abs(np.degrees(np.arctan(np.divide(self.RadarTracks[idx]['EgoMeasZ'],self.RadarTracks[idx]['EgoMeasX'])))) #will always be negative, so correct for it
    #     self.CameraX=self.CameraAnglesH*(self.RawImage.shape[1]/self.H_FOV) + self.RawImage.shape[1]/2 +self.HorzOffset# Number of pixels per degree,adjusted for shifting origin from centerline to top left
    #     self.CameraY=self.CameraAnglesV*(self.RawImage.shape[0]/self.V_FOV) +256 +self.VertOffset -self.RadarTracks[idx]['EgoMeasX']*np.sin(np.radians(4)) # Number of pixels per degree,adjusted for shifting origin from centerline to top left
        


        
    #     imageTemp=cv2.cvtColor(imageTemp,cv2.COLOR_GRAY2RGB)
    #     for idx in range(len(self.CameraAnglesH)):
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
