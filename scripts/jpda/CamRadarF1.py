#!/usr/bin/env python
import rospy
import numpy as np
import sys
import os
import tf
from dbw_mkz_msgs.msg import SteeringReport
from sensor_msgs.msg import Image
from derived_object_msgs.msg import ObjectWithCovarianceArray
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from darknet_ros_msgs.msg import BoundingBoxes
from darknet_ros_msgs.msg import ObjectCount
from nuscenes2bag.msg import RadarObjects
from utils import Mat_buildROS
from utils import Mat_extractROS
from flir_adk_multi.msg import trackArrayRdr
from flir_adk_multi.msg import trackRdr
from flir_adk_multi.msg import trackArrayCam
from flir_adk_multi.msg import trackCam
from dbw_mkz_msgs.msg import WheelSpeedReport
from utils import CamObj
from utils import RadarObj
from utils import RadarObjMKZ
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError
import cv2
from itertools import permutations
import time

def main():
    rospy.init_node('jpda', anonymous=True)
    DataSetType=sys.argv[1]
    Method=sys.argv[2]
    PlotArg=sys.argv[3] # 0-No Plot; 1-Combined; 2-Cam; 3-Rdr; 4-Both Cam&Rdr
    fusInst=jpda_class(DataSetType,Method,PlotArg)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
class jpda_class():

    def __init__(self,DataSetType,Method,PlotArg):
        self.TrackPubRdr=rospy.Publisher("dataAssocRdr",trackArrayRdr, queue_size=2) 
        self.TrackPubCam=rospy.Publisher("dataAssocCam",trackArrayCam, queue_size=2) 
        self.image_pub=rospy.Publisher("fusedImage",Image, queue_size=2) 
        filePathPrefix=str("/home/vamsi/Tracking/py-motmetrics/motmetrics/res_dir/")
        self.DestF=open((filePathPrefix+'seq1Jul12Bag8'+'.txt'),"w")
        # self.YoloClassList=[0,1,2,3,5,7] # For NuSc
        self.YoloClassList=[0,1,2] # For Yolov3_flir
        self.GateThreshRdr =1# Scaling factor, threshold for gating
        self.GateThreshCam=15# TODO: adjust?
        self.trackInitRdrThresh=0.3 # For track initiation
        self.trackInitCamThresh=15 # Radius of 15 pixels allowed
        self.CombGateThresh=15# in pixels (added to radius buffer)
        self.bridge=CvBridge()
        self.font=cv2.FONT_HERSHEY_SIMPLEX 
        # Initializing parameters:
        self.Q_rdr=np.array([[10,0,0,0],[0,10,0,0],[0,0,5,0],[0,0,0,1]])
        self.R_rdr=np.array([[3,0,0],[0,3,0],[0,0,3]])
        # self.Q_cam=np.diag([10,10,15,15,10,10,15,15])
        self.Q_cam=np.diag([10,10,10,10,10,10,20,20])
        self.R_cam=np.array([[10,0,0,0],[0,10,0,0],[0,0,5,0],[0,0,0,5]])
        self.CamMsrtmMatrixH=np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],\
            [0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]]) # Only positions and w/h are measured)
        self.Vt=0.0
        self.velY=0.0
        self.velX=0.0
        self.psi=0.0
        self.psiD=0.0# psiDot
        self.Method=Method
        self.PlotArg=PlotArg
        self.HorzOffset=0# For translation from radar to cam coordinates, manual offset
        self.CamXOffset=2.36#=93 inches, measured b/w cam and Rdr, in x direction
        self.CamZoffset=1 # Roughly 40 inches
        self.imageTime=Header()
        self.BBoxStore=BoundingBoxes()
        
        #Params for writing tracks to TXT:
        self.delta_x = 0
        self.delta_y = 0 # Assuming that the radar and camera are on same centerline
        self.delta_z = 1.0414/2
        self.H_FOV=190
        self.V_FOV=41 #Calculated based on aspect ratio
        self.HorzOffsetTXT=0 # Manual horizontal (Y-direction) offset for radar in pixels
        self.VertOffsetTXT=-30 # Manual vertical (Z-direction) offset for radar in pixels
        self.ImageExists=0
        self.BBheight=90
        self.BBWidth=90 # For now, static
        self.FrameInit=1
        self.UseCamTracksOnly=1 #1 if using only camera tracks, 0 if using combined tracks for eval
        
        if DataSetType=="NuSc":
            rospy.Subscriber('/cam_front/raw', Image, self.buildImage)
            rospy.Subscriber('/vel', Twist, self.Odom1NuSc) 
            rospy.Subscriber('/odom', Odometry, self.Odom2NuSc) 
            rospy.Subscriber('/imu', Imu, self.Odom3NuSc)
            rospy.Subscriber('/radar_front', BoundingBoxes, self.RdrMsrmtsNuSc)
        elif DataSetType=="MKZ":
            self.CamFOV=190.0
            rospy.Subscriber('/Thermal_Panorama', Image, self.buildImage)
            rospy.Subscriber('/imu/data', Imu, self.Odom2MKZ) # TODO: fix after IMU is available
            rospy.Subscriber('/vehicle/twist', TwistStamped,self.Odom3MKZ)
            rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes,self.BBoxBuilder)
            rate=rospy.Rate(10) # 20 Hz
            while not rospy.is_shutdown():
                # CycleStartTime=time.time()
                # startTime=time.time()
                # rospy.Subscriber('/as_tx/objects', ObjectWithCovarianceArray,self.RdrMsrmtsMKZ)
                # rospy.Subscriber('/darknet_ros/found_object', ObjectCount,self.CamMsrmts)
                
                self.RdrMsrmtsMKZ(rospy.wait_for_message('/as_tx/objects', ObjectWithCovarianceArray))
                self.CamMsrmts(rospy.wait_for_message('/darknet_ros/found_object', ObjectCount))

                # # print('TOTAL for RDR:' + str(time.time()-startTime))
                
                # # startTime=time.time()
                # try:
                #    rospy.Subscriber('/darknet_ros/found_object', ObjectCount,self.CamMsrmts)
                # except:
                #     rospy.loginfo('No Camera Data/Bounding Boxes found')
                #     pass
                # print('TOTAL for CAM:' + str(time.time()-startTime))
                # startTimeCom=time.time()
                
                # print('Time Combining:' + str(time.time()-startTimeCom))
                # print('Total Cycle Time:' + str(time.time()-CycleStartTime))
                self.CamRdrCombine()
                rate.sleep()
                
        elif DataSetType=="matlab":
            self.CamFOV=50
            rospy.Subscriber('/Thermal_Panorama', Image, self.buildImage)
            rospy.Subscriber('/imu/data', Imu, self.Odom2MKZ) # TODO: fix after IMU is available
            rospy.Subscriber('/vehicle/twist', TwistStamped,self.Odom3MKZ)
            rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes,self.BBoxBuilder)
            
            rate=rospy.Rate(10) # 20 Hz
            while not rospy.is_shutdown():
                # CycleStartTime=time.time()
                # startTime=time.time()
                # rospy.Subscriber('/as_tx/objects', ObjectWithCovarianceArray,self.RdrMsrmtsMKZ)
                # rospy.Subscriber('/darknet_ros/found_object', ObjectCount,self.CamMsrmts)
                
                self.RdrMsrmtsMKZ(rospy.wait_for_message('/as_tx/objects', ObjectWithCovarianceArray))
                self.CamMsrmts(rospy.wait_for_message('/darknet_ros/found_object', ObjectCount))

                # # print('TOTAL for RDR:' + str(time.time()-startTime))
                
                # # startTime=time.time()
                # try:
                #    rospy.Subscriber('/darknet_ros/found_object', ObjectCount,self.CamMsrmts)
                # except:
                #     rospy.loginfo('No Camera Data/Bounding Boxes found')
                #     pass
                # print('TOTAL for CAM:' + str(time.time()-startTime))
                # startTimeCom=time.time()
                
                # print('Time Combining:' + str(time.time()-startTimeCom))
                # print('Total Cycle Time:' + str(time.time()-CycleStartTime))
                self.CamRdrCombine()
                rate.sleep()
            
        
    def buildImage(self,data):
        if not(hasattr(self,'image')):
            self.image=[]
        self.image=self.bridge.imgmsg_to_cv2(data, "rgb8")
        self.imageTime=data.header
        self.ImageExists=1
        self.writeToFile() #Only write to file everytime a new frame is published


    def Odom1NuSc(self,data):
        self.Vt =data.linear.x 
    def Odom2NuSc(self,data):
        self.psi=data.pose.pose.orientation.z
    def Odom3NuSc(self,data):
        self.psiD=data.angular_velocity.z
    def Odom1MKZ(self,data): # REMOVE
        self.Vt=data.speed
    def Odom2MKZ(self,data):
        self.psi=tf.transformations.euler_from_quaternion([data.orientation.x,data.orientation.y,data.orientation.z,data.orientation.w])[2]
        # psi above is in radians, with 0 facing due EAST, not north
        
    def Odom3MKZ(self,data):
        self.psiD=data.twist.angular.z
        self.Vt=data.twist.linear.x
        self.velX=self.Vt # For use in calculating velocity of cut in vehicle(tracking target), Vc 

    def writeToFile(self):
        if not hasattr(self,'CombinedTracks'):
            return
        # self.Readoings=[]
        # n=len(self.RadarTracks)
        RadarAnglesH=0.0
        RadarAnglesV=0.0
        frame=self.FrameInit
        self.FrameInit+=1
        if self.UseCamTracksOnly==1:
            writeTracks=self.CurrentCamTracks
        else:
            writeTracks=self.CombinedTracks

        for idx in range(len(writeTracks.tracks)):
            # if (data.objects[idx].pose.pose.position.x==0.0) and (data.objects[idx].pose.pose.position.y==0.0) and (data.objects[idx].pose.covariance[0]==0.0):
            #     continue #Zero entry, so skip it
            # else: #write to file
            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            
            id=int(idx+1) # TODO: This is temp, not true ID of car
            # RadarX=data.objects[idx].pose.pose.position.x+self.delta_x
            # RadarY=data.objects[idx].pose.pose.position.y
            # RadarZ=0.0+self.delta_z
            # RadarAnglesH=-np.degrees(np.arctan(np.divide(RadarY,RadarX)))
            # RadarAnglesV=np.abs(np.degrees(np.arctan(np.divide(RadarZ,RadarX)))) #will always be negative, so correct for it
            

            
            if self.ImageExists==1:
                # imageTemp = self.image
                # print(imageTemp.shape)
                # CameraX=RadarAnglesH*(self.image.shape[1]/self.H_FOV) + self.image.shape[1]/2 +self.HorzOffsetTXT# Number of pixels per degree,adjusted for shifting origin from centerline to top left
                # CameraY=RadarAnglesV*(self.image.shape[0]/self.V_FOV) +256 +self.VertOffsetTXT -RadarX*np.sin(np.radians(4)) # Number of pixels per degree,adjusted for shifting origin from centerline to top left
                #Write to File
                bb_left=int(writeTracks.tracks[idx].yPx.data)
                bb_top=int(writeTracks.tracks[idx].zPx.data)
                bb_width=int(writeTracks.tracks[idx].width.data)
                bb_height=int(writeTracks.tracks[idx].height.data)
                x=-1 # Fillers
                y=-1
                z=-1
                conf=1
                outLine=str(frame)+' '+str(id)+' '+str(bb_left)+' '+str(bb_top)+' '+str(bb_width)+' '+str(bb_height)+' '+str(conf)+' '+str(x)+' '+str(y)+' '+str(z)+'\n'
                # print(outLine)
                self.DestF.write(outLine)

    def CamIOUcheck(self,checkIdx):
        #Return boolean. checks if IOU of given SensorIndex over any Current tracks is greater than threshold
        # if it is, then returns false
        outBool2=True # By default
        #TODO: perform check if required
        return outBool2

    def trackInitiator(self,SensorData):
        if not any(SensorData):
            return
        elif isinstance(SensorData[0],CamObj): 
            if hasattr(self, 'InitiatedCamTracks'):
                # Then, move to current tracks based on NN-style gating
                
                toDel=[]
                InitiatedCamTracks=self.InitiatedCamTracks
                # first build array of all sensor indices that are within validation gate of current tracks
                if hasattr(self,'CurrentCamTracks'):
                    TempCurrTracks=self.CurrentCamTracks
                    SensorIndicesInit=[]
                    for cdx in range(len(TempCurrTracks.tracks)):
                        SensorIndicesInit.append(self.ValidationGate(SensorData,TempCurrTracks.tracks[cdx]))
                else:
                    SensorIndicesInit=[] 
                for idx in range(len(InitiatedCamTracks.tracks)):
                    R=[]
                    if len(SensorData)==0:
                        continue
                    for jdx in range(len(SensorData)):
                        # If the Sensor Data is already in validatation gate of any of the currentTracks, skip adding that into InitiatedTracks
                        if self.InitSensorValidator(SensorIndicesInit,jdx):
                            continue 
                        else:
                            R.append(np.sqrt((InitiatedCamTracks.tracks[idx].yPx.data-(SensorData[jdx].xmax+SensorData[jdx].xmin)/2)**2 \
                            +(InitiatedCamTracks.tracks[idx].zPx.data-(SensorData[jdx].ymax+SensorData[jdx].ymin)/2)**2))
                    if len(R)==0:
                        R=9000 #Arbitrarily large value
                    R=np.asarray(R)
                    # print()
                    # print(R)
                    if (np.min(R)<self.trackInitCamThresh): # Then move this to current track # Inherent assumption here is that only one will be suitable
                        jdx=np.argmin(R)
                        if  not hasattr(self, 'CurrentCamTracks'):
                            self.CurrentCamTracks=trackArrayCam()
                        delT=self.imageTime.stamp-InitiatedCamTracks.header.stamp
                        delT=delT.to_sec()
                        self.CurrentCamTracks.header=SensorData[jdx].header
                        InitiatedCamTracks.tracks[idx].Stat.data=1 # Moving to current track
                        # Update the track with new sensor data before pushing to Current tracks
                        InitiatedCamTracks.tracks[idx].VyPx.data=\
                            (InitiatedCamTracks.tracks[idx].yPx.data-(SensorData[jdx].xmax+SensorData[jdx].xmin)/2)/delT
                        InitiatedCamTracks.tracks[idx].VzPx.data=\
                            (InitiatedCamTracks.tracks[idx].zPx.data-(SensorData[jdx].ymax+SensorData[jdx].ymin)/2)/delT
                        InitiatedCamTracks.tracks[idx].widthDot.data=\
                            (InitiatedCamTracks.tracks[idx].width.data-(SensorData[jdx].xmax-SensorData[jdx].xmin))/delT
                        InitiatedCamTracks.tracks[idx].heightDot.data=\
                            (InitiatedCamTracks.tracks[idx].height.data-(SensorData[jdx].ymax-SensorData[jdx].ymin))/delT
                        InitiatedCamTracks.tracks[idx].height.data=(SensorData[jdx].ymax-SensorData[jdx].ymin)
                        InitiatedCamTracks.tracks[idx].width.data=(SensorData[jdx].xmax-SensorData[jdx].xmin)
                        InitiatedCamTracks.tracks[idx].yPx.data=(SensorData[jdx].xmax+SensorData[jdx].xmin)/2
                        InitiatedCamTracks.tracks[idx].zPx.data=(SensorData[jdx].ymax+SensorData[jdx].ymin)/2
                        InitiatedCamTracks.tracks[idx].confidence=SensorData[jdx].confidence
                        Pk=np.diag([5,5,5,5,50,50,50,50]) # Initial covariance matrix
                        InitiatedCamTracks.tracks[idx].P=Mat_buildROS(Pk)
                        self.CurrentCamTracks.tracks=np.append(self.CurrentCamTracks.tracks,InitiatedCamTracks.tracks[idx])
                        toDel.append(idx)
                        SensorData=np.delete(SensorData,np.argmin(R))
                        
                    else: # for this idx of InitiatedCamTrack, the last jdx, so no measurements are nearby; delete the idx
                        toDel.append(idx)
                #Clean all InitiatedCamTracks using toDel
                self.InitiatedCamTracks.tracks=np.delete(InitiatedCamTracks.tracks,toDel)
                #Remove old initiated tracks (if idle for more than 3 time steps):
                toDel2=[]
                for idx in range(len(self.InitiatedCamTracks.tracks)):
                    self.InitiatedCamTracks.tracks[idx].Stat.data=self.InitiatedCamTracks.tracks[idx].Stat.data-1
                    if self.InitiatedCamTracks.tracks[idx].Stat.data<0:
                        toDel2.append(idx)
                self.InitiatedCamTracks.tracks=np.delete(self.InitiatedCamTracks.tracks,toDel2)
                # Then concatenate remaining sensor Data for future initation
                if len(SensorData)==0:
                    return
                self.InitiatedCamTracks.header=SensorData[0].header
                for idx in range(len(SensorData)):
                    self.InitiatedCamTracks.tracks=np.append(self.InitiatedCamTracks.tracks,trackCam())
                    self.InitiatedCamTracks.tracks[-1].Stat.data= -1 # InitiatedTrack
                    self.InitiatedCamTracks.tracks[-1].yPx.data=(SensorData[idx].xmax+SensorData[idx].xmin)/2
                    self.InitiatedCamTracks.tracks[-1].zPx.data=(SensorData[idx].ymax+SensorData[idx].ymin)/2
                    self.InitiatedCamTracks.tracks[-1].VyPx.data=0
                    self.InitiatedCamTracks.tracks[-1].VzPx.data=0
                    self.InitiatedCamTracks.tracks[-1].width.data=(SensorData[idx].xmax-SensorData[idx].xmin)
                    self.InitiatedCamTracks.tracks[-1].widthDot.data=0
                    self.InitiatedCamTracks.tracks[-1].height.data=(SensorData[idx].ymax-SensorData[idx].ymin)
                    self.InitiatedCamTracks.tracks[-1].heightDot.data=0
                    self.InitiatedCamTracks.tracks[-1].confidence=SensorData[idx].confidence

            else: # Start of algorithm, no tracks
                self.InitiatedCamTracks=trackArrayCam()
                self.InitiatedCamTracks.header=SensorData[0].header
                for idx in range(len(SensorData)):
                    self.InitiatedCamTracks.tracks=np.append(self.InitiatedCamTracks.tracks,trackCam())
                    self.InitiatedCamTracks.tracks[-1].Stat.data=-1 # Initiated Track
                    self.InitiatedCamTracks.tracks[-1].yPx.data=(SensorData[idx].xmax+SensorData[idx].xmin)/2
                    self.InitiatedCamTracks.tracks[-1].zPx.data=(SensorData[idx].ymax+SensorData[idx].ymin)/2
                    self.InitiatedCamTracks.tracks[-1].VyPx.data=0
                    self.InitiatedCamTracks.tracks[-1].VzPx.data=0
                    self.InitiatedCamTracks.tracks[-1].width.data=(SensorData[idx].xmax-SensorData[idx].xmin)
                    self.InitiatedCamTracks.tracks[-1].widthDot.data=0
                    self.InitiatedCamTracks.tracks[-1].height.data=(SensorData[idx].ymax-SensorData[idx].ymin)
                    self.InitiatedCamTracks.tracks[-1].heightDot.data=0
                    self.InitiatedCamTracks.tracks[-1].confidence=SensorData[idx].confidence
                
            
        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
            if hasattr(self, 'InitiatedRdrTracks'):# Some (or Zer0) tracks already exists (i.e, not start of algorithm)
                toDel=[]
                InitiatedRdrTracks=self.InitiatedRdrTracks
                # first build array of all sensor indices that are within validation gate of current tracks
                if hasattr(self,'CurrentRdrTracks'):
                    TempCurrTracksRdr=self.CurrentRdrTracks
                    SensorIndicesInitRdr=[]
                    for cdx in range(len(TempCurrTracksRdr.tracks)):
                        SensorIndicesInitRdr.append(self.ValidationGate(SensorData,TempCurrTracksRdr.tracks[cdx]))
                else:
                    SensorIndicesInitRdr=[] 
                for idx in range(len(self.InitiatedRdrTracks.tracks)):
                    gateValX=[]
                    gateValY=[]
                    gateValRMS=[]
                    # Find all sensor objects within some gate
                    if len(SensorData)==0:
                        continue
                    for jdx in range(len(SensorData)):
                        if self.InitSensorValidator(SensorIndicesInitRdr,jdx):
                            continue 
                        else:
                            gateValX.append(np.abs(SensorData[jdx].pose.position.x-self.InitiatedRdrTracks.tracks[idx].x.data))
                            gateValY.append(np.abs(SensorData[jdx].pose.position.y-self.InitiatedRdrTracks.tracks[idx].y.data))
                            gateValRMS.append(np.sqrt((gateValX[-1])**2+(gateValY[-1])**2))
                    if len(gateValRMS)==0:
                        gateValRMS=1000# Arbitrary large value, greater than trackInitRdrThresh
                    if (np.min(np.array(gateValRMS))<=self.trackInitRdrThresh): # @50Hz, 20m/s in X dir and 10m/s in Y-Direction as validation gate
                        #If gate is satisfied, move to CurrentRdrTracks after initiating P and deleting that SensorData[idx]
                        self.InitiatedRdrTracks.tracks[idx].P=Mat_buildROS(np.array([[3,0,0,0],[0,3,0,0],[0,0,3,0],[0,0,0,1]]))
                        #(Large uncertainity given to Beta. Others conservatively picked based on Delphi ESR spec sheet)
                        self.InitiatedRdrTracks.tracks[idx].Stat.data=1# Moving to CurrentRdrTracks
                        x=self.InitiatedRdrTracks.tracks[idx].x.data
                        y=self.InitiatedRdrTracks.tracks[idx].y.data
                        Vc=self.InitiatedRdrTracks.tracks[idx].Vc.data
                        Beta=self.InitiatedRdrTracks.tracks[idx].B.data
                        psi=np.array([self.psi])
                        psiD=np.array([self.psiD])
                        Vt=self.Vt
                        posNorm=np.sqrt(x**2+y**2)
                        H31=(Vc*np.sin((psi-Beta).astype(float))*y**2-x*y*(Vc*np.cos((psi-Beta).astype(float))-Vt))/(posNorm**3)
                        H32=(-Vc*np.sin((psi-Beta).astype(float))*x*y+x**2*(Vc*np.cos((psi-Beta).astype(float))-Vt))/(posNorm**3)
                        H33=x*np.sin((psi-Beta).astype(float))/posNorm+y*np.cos((psi-Beta).astype(float))/posNorm
                        H34=(-x*Vc*np.cos((psi-Beta).astype(float))+y*Vc*np.sin((psi-Beta).astype(float)))/posNorm
                        Hk=np.array([[1,0,0,0],[x/posNorm,y/posNorm,0,0],[H31,H32,H33,H34]])
                        self.InitiatedRdrTracks.tracks[idx].H=Mat_buildROS(Hk)
                        if hasattr(self, 'CurrentRdrTracks'):
                            pass
                        else:
                            self.CurrentRdrTracks=trackArrayRdr()
                        self.CurrentRdrTracks.header=self.InitiatedRdrTracks.header
                        self.CurrentRdrTracks.tracks=np.append(self.CurrentRdrTracks.tracks,self.InitiatedRdrTracks.tracks[idx])
                        #Build Arrays for deletion:
                        toDel.append(idx)
                        #Also Delete the corresponding SensorData value:
                        SensorData=np.delete(SensorData,np.argmin(gateValRMS))
                    else: # none of the SensorData is close to InitiatedRdrTracks[idx], so delete it
                        toDel.append(idx)
                # Clean all InitiatedRdrTracks with status 1
                self.InitiatedRdrTracks.tracks=np.delete(self.InitiatedRdrTracks.tracks,toDel)
                #Remove old initiated tracks:(if idle for more than 2 time steps):
                toDel2=[]
                for idx in range(len(self.InitiatedRdrTracks.tracks)):
                    self.InitiatedRdrTracks.tracks[idx].Stat.data=self.InitiatedRdrTracks.tracks[idx].Stat.data-1
                    if self.InitiatedRdrTracks.tracks[idx].Stat.data<=-3:
                        toDel2.append(idx)
                self.InitiatedRdrTracks.tracks=np.delete(self.InitiatedRdrTracks.tracks,toDel2)
                # Then concatenate remaining sensor Data for future initation
                if len(SensorData)==0:
                    return
                self.InitiatedRdrTracks.header=SensorData[0].header
                for idx in range(len(SensorData)):
                    self.InitiatedRdrTracks.tracks=np.append(self.InitiatedRdrTracks.tracks,trackRdr())
                    self.InitiatedRdrTracks.tracks[-1].Stat.data= -1 # InitiatedTrack
                    self.InitiatedRdrTracks.tracks[-1].x.data=SensorData[idx].pose.position.x
                    self.InitiatedRdrTracks.tracks[-1].y.data=SensorData[idx].pose.position.y
                    self.InitiatedRdrTracks.tracks[-1].Vc.data=np.sqrt(SensorData[idx].vx_comp**2+SensorData[idx].vy_comp**2)
                    self.InitiatedRdrTracks.tracks[-1].B.data=self.psi -(np.arctan(SensorData[idx].pose.position.y/(0.0001 if (SensorData[idx].pose.position.x)==0.0 else (SensorData[idx].pose.position.x))))
                    # TODO: Improve Beta estimate by taking into account relative Vx(invert heading if object istraveling towards car)
                    

            else: # Start of algorithm, no tracks
                self.InitiatedRdrTracks=trackArrayRdr()
                self.InitiatedRdrTracks.header=SensorData[0].header
                for idx in range(len(SensorData)):
                    self.InitiatedRdrTracks.tracks=np.append(self.InitiatedRdrTracks.tracks,trackRdr())
                    self.InitiatedRdrTracks.tracks[-1].Stat.data= -1 # InitiatedTrack
                    self.InitiatedRdrTracks.tracks[-1].x.data=SensorData[idx].pose.position.x
                    self.InitiatedRdrTracks.tracks[-1].y.data=SensorData[idx].pose.position.y
                    self.InitiatedRdrTracks.tracks[-1].Vc.data=np.sqrt(SensorData[idx].vx_comp**2+SensorData[idx].vy_comp**2)
                    self.InitiatedRdrTracks.tracks[-1].B.data=self.psi -(np.arctan(SensorData[idx].pose.position.y/(0.0001 if (SensorData[idx].pose.position.x)==0.0 else (SensorData[idx].pose.position.x))))
                    # TODO: Improve Beta estimate by taking into account relative Vx(invert heading if object istraveling towards car)

    def trackDestructor(self,SensorData):
        if not any(SensorData):
            return
        if isinstance(SensorData[0],CamObj): 
            if not (hasattr(self,'CurrentCamTracks')):
                return
            toDel=[]
            for idx in range(len(self.CurrentCamTracks.tracks)):
                if self.CurrentCamTracks.tracks[idx].Stat.data>=2:# Testing, made less persistent
                    toDel.append(idx)
            self.CurrentCamTracks.tracks=np.delete(self.CurrentCamTracks.tracks,toDel)
        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
            if not(hasattr(self,'CurrentRdrTracks')):
                return
            toDel=[]
            for idx in range(len(self.CurrentRdrTracks.tracks)):
                if self.CurrentRdrTracks.tracks[idx].Stat.data>=4: # If no measurements associated for 4 steps
                    toDel.append(idx)
            self.CurrentRdrTracks.tracks=np.delete(self.CurrentRdrTracks.tracks,toDel)

    def trackMaintenance(self,SensorData):
        if not any(SensorData):
            return
        if isinstance(SensorData[0],CamObj): 
            if  not hasattr(self, 'CurrentCamTracks'):
                return
            SensorIndices=[]
            for idx in range(len(self.CurrentCamTracks.tracks)):
                SensorIndices.append(self.ValidationGate(SensorData,self.CurrentCamTracks.tracks[idx]))#Clean the incoming data - outputs 2D python array
                # Above yields array of possible measurments (only indices) corresponding to a particular track
            # startTime1=time.time()    
            self.KalmanEstimate(SensorData,SensorIndices, self.Method) # Includes DataAssociation Calcs
            # print('Time for KalmanEstimate:' + str(time.time()-startTime1))
            # startTime2=time.time() 
            self.KalmanPropagate(SensorData)
            # print('Time for KalmanPropagate:' + str(time.time()-startTime2))
            self.TrackPubCam.publish(self.CurrentCamTracks)
        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
            if  not hasattr(self, 'CurrentRdrTracks'):
                return
            SensorIndices=[]
            for idx in range(len(self.CurrentRdrTracks.tracks)):
                SensorIndices.append(self.ValidationGate(SensorData,self.CurrentRdrTracks.tracks[idx]))#Clean the incoming data - outputs 2D python array
                # Above yields array of possible measurments (only indices) corresponding to a particular track
            # startTimeKE=time.time()    
            self.KalmanEstimate(SensorData,SensorIndices, self.Method) # Includes DataAssociation Calcs
            # print('Time for KalmanEstimate:' + str(time.time()-startTimeKE))
            # startTimeKP=time.time() 
            self.KalmanPropagate(SensorData)
            # print('Time for KalmanPropagate:' + str(time.time()-startTimeKP))
            # self.TrackPubRdr.publish(header=self.CurrentRdrTracks.header, tracks =self.CurrentRdrTracks.tracks)
            # rospy.loginfo_once('Current tracks published to topic /dataAssoc')

    def InitSensorValidator(self,SensorIndicesInit,jdx):
        #takes SensorIndices 2 D python array and current Sensor index being checked;
        # returns true if the current index is in the 2D array
        outBool=False # By default
        if len(SensorIndicesInit)==0:
            return outBool
        for sens_idx in range(len(SensorIndicesInit)):
            if jdx in SensorIndicesInit[sens_idx]:
                outBool=True
        return outBool


    def trackPlotter(self):
        if not (hasattr(self,'image')) or (self.PlotArg=='0'):
           return # Skip function call if image is not available or plotting is disabled
        LocalImage=self.image
        if (self.PlotArg=='3') or (self.PlotArg=='4'): # Then, plot Radar stuff
            if not hasattr(self,'CurrentRdrTracks'):
                return # Skip
            CurrentRdrTracks=self.CurrentRdrTracks
            n=len(CurrentRdrTracks.tracks)
            RadarAnglesH=np.zeros((n,1))
            RadarAnglesV=np.zeros((n,1))
            # Camera Coordinates: X is horizontal, Y is vertical starting from left top corner
            CirClr=[]
            for idx1 in range(len(CurrentRdrTracks.tracks)):
                temp1=np.divide(CurrentRdrTracks.tracks[idx1].y.data,CurrentRdrTracks.tracks[idx1].x.data)
                RadarAnglesH[idx1]=-np.degrees(np.arctan(temp1.astype(float)))
                temp2=np.divide(self.CamZoffset,CurrentRdrTracks.tracks[idx1].x.data+self.CamXOffset)
                RadarAnglesV[idx1]=np.abs(np.degrees(np.arctan(temp2.astype(float)))) #will always be negative, so correct for it
                if (CurrentRdrTracks.tracks[idx1].Stat.data>=1) and (CurrentRdrTracks.tracks[idx1].Stat.data<14): #Current Track- green
                    CirClr.append(np.array([0,255,0]))
                elif CurrentRdrTracks.tracks[idx1].Stat.data<=0: # Candidate Tracks for initialization - blue
                    CirClr.append(np.array([255,0,0]))
                else: # Candidate for Destructor-orange
                    CirClr.append(np.array([0,165,255])) 
            CameraX=np.dot(RadarAnglesH,(self.image.shape[1]/self.CamFOV)) + self.image.shape[1]/2 # Number of pixels per degree,adjusted for shifting origin from centerline to top left
            CameraY=np.dot(RadarAnglesV,(self.image.shape[0]/(39.375))) +480/2 # Number of pixels per degree,adjusted for shifting origin from centerline to top left
            CirClr=np.array(CirClr)
            CameraX=np.array(CameraX)
            for idx3 in range(len(CameraX)):
                if (CameraX[idx3]<=self.image.shape[1]):
                    LocalImage=cv2.circle(LocalImage, (int(CameraX[idx3]),int(CameraY[idx3])), 12, CirClr[idx3].tolist(),3)
                    LocalImage=cv2.putText(LocalImage,str(idx3),(int(CameraX[idx3]),int(CameraY[idx3])),self.font,1,(255,105,180),2)
        
        #Now Plot Camera Trakcs:
        if (self.PlotArg=='2') or (self.PlotArg=='4'): # Then, plot Cam stuff
            if not hasattr(self,'CurrentCamTracks'):
                return # Skip
            CurrentCamTracks=self.CurrentCamTracks
            RectClr=[]
            for jdx in range(len(CurrentCamTracks.tracks)):
                if (CurrentCamTracks.tracks[jdx].Stat.data>=1) and (CurrentCamTracks.tracks[jdx].Stat.data<14): #Current Track- green
                    RectClr.append(np.array([0,255,0]))
                elif CurrentCamTracks.tracks[jdx].Stat.data<=0: # Candidate Tracks for initialization - blue
                    RectClr.append(np.array([255,0,0]))
                else: # Candidate for Destructor-orange
                    RectClr.append(np.array([0,165,255])) 
            for idx2 in range(len(CurrentCamTracks.tracks)):
                start=(int(CurrentCamTracks.tracks[idx2].yPx.data-CurrentCamTracks.tracks[idx2].width.data/2),int(CurrentCamTracks.tracks[idx2].zPx.data-CurrentCamTracks.tracks[idx2].height.data/2))
                end= (int(CurrentCamTracks.tracks[idx2].yPx.data+CurrentCamTracks.tracks[idx2].width.data/2),int(CurrentCamTracks.tracks[idx2].zPx.data+CurrentCamTracks.tracks[idx2].height.data/2))
                LocalImage=cv2.rectangle(LocalImage,start,end,RectClr[idx2].tolist(),2)
        
        if (self.PlotArg=='1') or (self.PlotArg=='4'): # Only plot self.CombinedTracks
            if not hasattr(self,'CombinedTracks'):
                return
            currCombinedTracks=self.CombinedTracks
            RectClr=[]
            for jdx in range(len(currCombinedTracks.tracks)):
                    RectClr.append(np.array([102,255,255])) # Yellow 
            for idx2 in range(len(currCombinedTracks.tracks)):
                start=(int(currCombinedTracks.tracks[idx2].yPx.data-currCombinedTracks.tracks[idx2].width.data/2),int(currCombinedTracks.tracks[idx2].zPx.data-currCombinedTracks.tracks[idx2].height.data/2))
                end= (int(currCombinedTracks.tracks[idx2].yPx.data+currCombinedTracks.tracks[idx2].width.data/2),int(currCombinedTracks.tracks[idx2].zPx.data+currCombinedTracks.tracks[idx2].height.data/2))
                LocalImage=cv2.rectangle(LocalImage,start,end,RectClr[idx2].tolist(),2)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(LocalImage, "bgr8"))
        rospy.loginfo_once('Image is being published')
   
    def CamRdrCombine(self):
        if not hasattr(self,'CurrentCamTracks') or (not hasattr(self,'CurrentRdrTracks')):
            return
        self.CombinedTracks=trackArrayCam()
        n=len(self.CurrentCamTracks.tracks)
        LocalRdrYArr=[]
        for rdx in range(len(self.CurrentRdrTracks.tracks)):
            temp1=np.divide(self.CurrentRdrTracks.tracks[rdx].y.data,self.CurrentRdrTracks.tracks[rdx].x.data)
            temp2=-np.degrees(np.arctan(temp1.astype(float)))
            LocalRdrYArr.append(np.dot(temp2,(self.image.shape[1]/self.CamFOV)) + self.image.shape[1]/2+self.HorzOffset) # Gives all Y-coord (pixels)  of all radar tracks 
        for jdx in range(n):
            radius=(self.CurrentCamTracks.tracks[jdx].width.data+self.CurrentCamTracks.tracks[jdx].height.data)/2+self.CombGateThresh
            centerY=self.CurrentCamTracks.tracks[jdx].yPx.data
            for Rdx in range(len(LocalRdrYArr)):
                if (abs(LocalRdrYArr[Rdx]-centerY)<=radius) or (self.CurrentCamTracks.tracks[jdx].confidence>=0.36):
                    self.CurrentCamTracks.tracks[jdx].Stat.data=99 #To indicate that the status is combined/validated
                    #TODO: Create a custom CombinedTracks Message that has both radar and Camera info?
                    self.CombinedTracks.tracks.append(self.CurrentCamTracks.tracks[jdx])
                    break
                else:
                    continue

    def trackManager(self,SensorData):
        # startTime01=time.time()
        self.trackMaintenance(SensorData)
        # print('Time for Track Maint:' + str(time.time()-startTime01))
        # startTime02=time.time()
        self.trackInitiator(SensorData)
        # print('Time for Track Init:' + str(time.time()-startTime02))
        # startTime03=time.time()
        self.trackDestructor(SensorData)
        # print('Time for Track Destr:' + str(time.time()-startTime03))
        # startTime04=time.time()
        self.trackPlotter()
        
        # print('Time for Track Plotter:' + str(time.time()-startTime04))
        # startTime05=time.time()
        if hasattr(self,'CurrentCamTracks') or hasattr(self,'CurrentRdrTracks'):
            s= '# Cam Tracks: ' + (str(len(self.CurrentCamTracks.tracks)) if hasattr(self,'CurrentCamTracks') else 'None') + \
             '; Rdr Tracks: ' + (str(len(self.CurrentRdrTracks.tracks)) if hasattr(self,'CurrentRdrTracks') else 'None') +'; # Combined Tracks:'\
            +(str(len(self.CombinedTracks.tracks)) if hasattr(self,'CombinedTracks') else 'None')
            print(s)
        # print('Time printing in track manager:' + str(time.time()-startTime05))
       
        

                
    def DataAssociation(self,SensorData,SensorIndices,Method):
        if Method=="Hungarian":
            pass
        elif Method=="JPDA":
            #Build A Validation Matrix if there are sufficient sensor data and tracks
            if (len(SensorData)<1) or (len(self.CurrentRdrTracks.tracks)<1):
                Yk=[]
            else:
                Yk=[]
                #create empty Yk list, with given number of targets (currentTracks):
                for l_dx in range(len(self.CurrentRdrTracks.tracks)):
                    Yk.append([])
                C=3 # Number of false measurements per unit volume (assume), clutter density
                Pd=0.9 #Probability of detection

                # Create Clusters by cycling through SensorIndices, maintain
                OpenList=[]
                ClusterList=[]
                for tempdx in range(len(self.CurrentRdrTracks.tracks)):
                    OpenList.append(tempdx)
                OpenList=np.array(OpenList)
                while any(OpenList):
                    tempClusterList=[]
                    tempClusterList.append(OpenList[0])
                    SensorRdgList=np.array(SensorIndices[OpenList[0]]).flatten()
                    OpenList=np.delete(OpenList,0) # Remove this element from searchable list of tracks, will be added later to ClusterList
                    # Chase down all other tracks that share common sensor measurements
                    n_meas=len(SensorData) # Total number of possible measurements
                    for m_dx in range(n_meas):
                        if m_dx in SensorRdgList:
                            ToDelOpenList=[]
                            for cluster_dx in OpenList:
                                indices = [i for i, obj in enumerate(SensorIndices[cluster_dx]) if obj == m_dx]
                                if any(indices) and (not (cluster_dx in tempClusterList)) :
                                    tempClusterList.append(cluster_dx)
                                    ToDelOpenList.append(cluster_dx) # To be Deleted from OpenList
                                    np.append(SensorRdgList,SensorIndices[cluster_dx]).flatten()
                            OpenList=np.setdiff1d(OpenList,ToDelOpenList) # Remove from OpenList
                        else:
                            continue
                    # Now add this cluster to ClusterList
                    ClusterList.append(tempClusterList)

                ### Directly calculate Bjt if cluster size is 1:4 as per Bose Paper
                # First calculate Yjt and Sjt:
                for tdx in range(len(self.CurrentRdrTracks.tracks)):
                    # Calculate Y_jt and S_jt
                    # First Sjt, since it only depends on t, not j
                    Sjt=np.zeros((len(self.CurrentRdrTracks.tracks),3,3))
                    Hk=Mat_extractROS(self.CurrentRdrTracks.tracks[tdx].H)
                    Pk=Mat_extractROS(self.CurrentRdrTracks.tracks[tdx].P)
                    Sjt[tdx]=np.matmul(np.matmul(Hk,Pk),Hk.T)+self.R_rdr
                def PjtCalc(meas_dx,target_dx,YjtLocal,Sjt):
                    if  meas_dx in SensorIndices[target_dx]:
                        Pjt=Pd*np.exp(-np.matmul(np.matmul(YjtLocal[:,meas_dx].T,Sjt[target_dx]),YjtLocal[:,meas_dx])/2)/(np.sqrt((2*np.pi)*np.linalg.det(Sjt[target_dx])))
                    else:
                        Pjt=0
                    return Pjt
                def GjCal(meas_dx,target_dx1, target_dx2,YjtLocal,Sjt):
                    Gj=PjtCalc(meas_dx,target_dx1,YjtLocal,Sjt)*PjtCalc(meas_dx,target_dx2,YjtLocal,Sjt)
                    return Gj


                def YjtCalc(t_idx):
                    yt=np.array([self.CurrentRdrTracks.tracks[t_idx].x.data,self.CurrentRdrTracks.tracks[t_idx].y.data, \
                        self.CurrentRdrTracks.tracks[t_idx].Vc.data]).reshape(3,1)
                    Yjt=np.zeros((3,len(SensorIndices[t_idx])))
                    for jdx in range(len(SensorIndices[t_idx])):
                        yjt=np.array([SensorData[SensorIndices[t_idx][jdx]].pose.position.x,SensorData[SensorIndices[t_idx][jdx]].pose.position.y, \
                            np.sqrt(SensorData[SensorIndices[t_idx][jdx]].vx_comp**2+SensorData[SensorIndices[t_idx][jdx]].vy_comp**2)]).reshape(3,1)
                        Yjt[:,jdx]=(yjt-yt).reshape(3)
                    return Yjt
                
                for clusterItem in ClusterList:
                    if len(clusterItem)==1:
                        B0t=C*(1-Pd)
                        Yjt=YjtCalc(clusterItem[0])
                        c=B0t
                        if len(SensorIndices[clusterItem[0]])>0:
                            Z_temp=np.zeros_like(Yjt[:,0])
                            for j_idx in range(len(SensorIndices[clusterItem[0]])):
                                Bjt=PjtCalc(j_idx,clusterItem[0],Yjt,Sjt)
                                c=c+Bjt
                                Z_temp=Z_temp+Bjt*Yjt[:,j_idx]
                                
                            Yk[clusterItem[0]]=Z_temp/c
                        else: # No measurement associated with this particular object in clusterItem
                            pass # Already Yk[clusterItem[0]] =[] by default
                    elif len(clusterItem)==2:
                        P0=C*(1-Pd)
                        P1=P0 
                        P2=P0
                        #Build P1:
                        Yjt1=YjtCalc(clusterItem[0])
                        for jdx in range(len(SensorIndices[clusterItem[0]])):
                            P1=P1+PjtCalc(jdx,clusterItem[0],Yjt1,Sjt)
                        # Build P2:
                        Yjt2=YjtCalc(clusterItem[1])
                        
                        for jdx in range(len(SensorIndices[clusterItem[1]])):
                            P2=P2+PjtCalc(jdx,clusterItem[1],Yjt2,Sjt)
                        #  Now build Bjts:
                        B0t1=P0*P2
                        c1=B0t1
                        # Calculate Bjt1:
                        Z_temp=np.zeros_like(Yjt1[:,0])
                        for j_idx in range(len(SensorIndices[clusterItem[0]])):
                            Bjt1=PjtCalc(j_idx,clusterItem[0],Yjt1,Sjt)*(P2-PjtCalc(j_idx,clusterItem[1],Yjt1,Sjt))
                            c1=c1+Bjt1
                            
                            Z_temp=Z_temp+Bjt1*Yjt1[:,j_idx]
                        # Add to Yk:
                        Yk[clusterItem[0]]=Z_temp/c1
                        # Now Calculate Bjt2:
                        B0t2=P0*P1
                        c2=B0t2
                        Z_temp=np.zeros_like(Yjt2[:,0])
                        for j_idx in range(len(SensorIndices[clusterItem[1]])):
                            Bjt2=PjtCalc(j_idx,clusterItem[1],Yjt2,Sjt)*(P1-PjtCalc(j_idx,clusterItem[0],Yjt2,Sjt))
                            c2=c2+Bjt2
                            Z_temp=Z_temp+Bjt2*Yjt2[:,j_idx]
                         # Add to Yk:
                        Yk[clusterItem[1]]=Z_temp/c1
                    elif len(clusterItem)==2:
                        # Build P's:
                        P0=C*(1-Pd)
                        P1=P0 
                        P2=P0
                        P3=P0
                        #Build P1:
                        Yjt1=YjtCalc(clusterItem[0])
                        for jdx in range(len(SensorIndices[clusterItem[0]])):
                            P1=P1+PjtCalc(jdx,clusterItem[0],Yjt1,Sjt)
                        # Build P2:
                        Yjt2=YjtCalc(clusterItem[1])
                        for jdx in range(len(SensorIndices[clusterItem[1]])):
                            P2=P2+PjtCalc(jdx,clusterItem[1],Yjt2,Sjt)
                        # Build P3:
                        Yjt3=YjtCalc(clusterItem[2])
                        for jdx in range(len(SensorIndices[clusterItem[2]])):
                            P3=P3+PjtCalc(jdx,clusterItem[2],Yjt3,Sjt)
                        # Now Build G's:
                        G23=0
                        for jdx in range(len(SensorIndices[clusterItem[0]])):
                            G23=G23+GjCal(jdx,1,2,Yjt1)
                        G13=0
                        for jdx in range(len(SensorIndices[clusterItem[1]])):
                            G13=G13+GjCal(jdx,0,2,Yjt2)
                        G12=0
                        for jdx in range(len(SensorIndices[clusterItem[2]])):
                            G12=G12+GjCal(jdx,0,1,Yjt3)

                        # Now Build Bjt's:
                        B0t1=P0*(P2*P3-G23)
                        c1=B0t1
                        B0t2=P0*(P1*P3-G13)
                        c2=B0t2
                        B0t3=P0*(P1*P2-G12)
                        c3=B0t3

                        Z_temp=np.zeros_like(Yjt1[:,0])
                        for j_idx in range(len(SensorIndices[clusterItem[0]])):
                            Bjt1=PjtCalc(j_idx,0,Yjt1,Sjt)*((P2-PjtCalc(j_idx,1,Yjt2,Sjt))*(P3-PjtCalc(meas_dx,2,Yjt3,Sjt))\
                                -(G23-GjCal(j_idx,1,2,Yjt1,Sjt)))
                            c1=c1+Bjt1
                            Z_temp=Z_temp+Bjt1*Yjt1[:,j_idx]
                        Yk[clusterItem[0]]=Z_temp/c1
                        Z_temp=np.zeros_like(Yjt2[:,0])
                        for j_idx in range(len(SensorIndices[clusterItem[0]])):
                            Bjt2=PjtCalc(j_idx,1,Yjt2,Sjt)*((P1-PjtCalc(j_idx,0,Yjt1,Sjt))*(P3-PjtCalc(meas_dx,2,Yjt3,Sjt))\
                                -(G13-GjCal(j_idx,0,2,Yjt2,Sjt)))
                            c2=c2+Bjt2
                            Z_temp=Z_temp+Bjt2*Yjt2[:,j_idx]
                        Yk[clusterItem[1]]=Z_temp/c2
                        Z_temp=np.zeros_like(Yjt3[:,0])
                        for j_idx in range(len(SensorIndices[clusterItem[0]])):
                            Bjt3=PjtCalc(j_idx,2,Yjt3,Sjt)*((P1-PjtCalc(j_idx,0,Yjt1,Sjt))*(P2-PjtCalc(meas_dx,1,Yjt2,Sjt))\
                                -(G12-GjCal(j_idx,0,1,Yjt3,Sjt)))
                            c3=c3+Bjt3
                            Z_temp=Z_temp+Bjt3*Yjt3[:,j_idx]
                        Yk[clusterItem[2]]=Z_temp/c3
                    
                    # If cluster size is greater than 4, use approximation as per paper (TODO, if required)
                    else:
                        print('Large Cluster Density, Skipping Data Association!!')
                        pass
                

            return Yk


                
        elif Method=="Greedy": # Simple method that just outputs the closest UNUSED measurement
            if isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
                # Sensor indices is a 2D python list, not numpy array
                usedSensorIndices=[]
                Yk=[] # A python list of sensor measurements corresponding to each CurrentTrack
                for idx in range(len(self.CurrentRdrTracks.tracks)):
                    gateValX=[]
                    gateValY=[]
                    gateValRMS=[]
                    if len(SensorIndices[idx])==0:
                        Yk.append([])
                        continue
                    else:
                        # print(len(SensorIndices[idx]))
                        for jdx in range(len(SensorIndices[idx])):
                            gateValX.append(np.abs(SensorData[SensorIndices[idx][jdx]].pose.position.x-self.CurrentRdrTracks.tracks[idx].x.data))
                            gateValY.append(np.abs(SensorData[SensorIndices[idx][jdx]].pose.position.y-self.CurrentRdrTracks.tracks[idx].y.data))
                            gateValRMS.append(np.sqrt(((gateValX[jdx])**2+(gateValY[jdx])**2).astype(float)))
                        if np.min(gateValRMS)<=self.GateThreshRdr:
                            sensIdx=int(np.argmin(np.array(gateValRMS)))
                            gateValRMS=np.array(gateValRMS)
                            temp=SensorData[sensIdx]
                            while sensIdx in usedSensorIndices:
                                gateValRMS=np.delete(gateValRMS,sensIdx)
                                if len(gateValRMS)==0:
                                    temp=[]
                                    break
                                sensIdx=int(np.argmin(np.array(gateValRMS)))
                                temp=SensorData[sensIdx]
                            usedSensorIndices.append(sensIdx)
                            Yk.append(temp)
                        else:
                            Yk.append([])
            elif isinstance(SensorData[0],CamObj): # Silimar To Radar above, gives closest unused sensor index
                # Sensor indices is a 2D python list, not numpy array
                usedSensorIndices=[]
                Yk=[] # A python list of sensor measurements corresponding to each CurrentTrack
                for idx in range(len(self.CurrentCamTracks.tracks)):
                    gateValX=[]
                    gateValY=[]
                    gateValRMS=[]
                    if len(SensorIndices[idx])==0:
                        Yk.append([])
                        continue
                    else:
                        # print(len(SensorIndices[idx]))
                        for jdx in range(len(SensorIndices[idx])):
                            gateValX.append(np.abs((SensorData[SensorIndices[idx][jdx]].xmin+SensorData[SensorIndices[idx][jdx]].xmax)/2-self.CurrentCamTracks.tracks[idx].yPx.data))
                            gateValY.append(np.abs((SensorData[SensorIndices[idx][jdx]].ymin+SensorData[SensorIndices[idx][jdx]].ymax)/2-self.CurrentCamTracks.tracks[idx].zPx.data))
                            gateValRMS.append(np.sqrt(((gateValX[jdx])**2+(gateValY[jdx])**2).astype(float)))
                        if np.min(gateValRMS)<=self.GateThreshCam:
                            sensIdx=int(np.argmin(np.array(gateValRMS)))
                            gateValRMS=np.array(gateValRMS)
                            temp=SensorData[sensIdx]
                            while sensIdx in usedSensorIndices:
                                gateValRMS=np.delete(gateValRMS,sensIdx)
                                if len(gateValRMS)==0:
                                    temp=[]
                                    break
                                sensIdx=int(np.argmin(np.array(gateValRMS)))
                                temp=SensorData[sensIdx]
                            usedSensorIndices.append(sensIdx)
                            Yk.append(temp)
                        else:
                            Yk.append([])
                # Yk=[]
                # for idx in range(len(self.CurrentCamTracks.tracks)):
                #     if len(SensorIndices[idx])==0:
                #         Yk.append([])
                #         continue
                #     else:
                #         Yk.append(SensorData[SensorIndices[idx][0]])

        return Yk # An Array with same len as CurrentRdrTracks.tracks[]

    def KalmanPropagate(self,SensorData):
        
        if isinstance(SensorData[0],CamObj) and  hasattr(self,'CurrentRdrTracks'): # TODO: hasattr part is just so we can get delT, this needs to be fixed
            delT=(SensorData[0].header.stamp-self.CurrentRdrTracks.header.stamp) 
            delT=delT.to_sec()
            for idx in range(len(self.CurrentCamTracks.tracks)):
                Fk=np.eye(8)
                Fk[0,4]=delT
                Fk[1,5]=delT
                Fk[2,6]=delT
                Fk[3,7]=delT
                self.CurrentCamTracks.tracks[idx].P=Mat_buildROS(Fk*Mat_extractROS(self.CurrentCamTracks.tracks[idx].P)*Fk.T+self.Q_cam)
                track=self.CurrentCamTracks.tracks[idx]
                StateVec=np.array([track.yPx.data,track.zPx.data,track.width.data,track.height.data,track.VyPx.data,\
                        track.VzPx.data,track.widthDot.data,track.heightDot.data])
                StateVec=np.dot(Fk,StateVec.reshape(8,1))
                self.CurrentCamTracks.tracks[idx].yPx.data=StateVec[0]
                self.CurrentCamTracks.tracks[idx].zPx.data=StateVec[1]
                self.CurrentCamTracks.tracks[idx].width.data=StateVec[2]
                self.CurrentCamTracks.tracks[idx].height.data=StateVec[3]
                self.CurrentCamTracks.tracks[idx].VyPx.data=StateVec[4]
                self.CurrentCamTracks.tracks[idx].VzPx.data=StateVec[5]
                self.CurrentCamTracks.tracks[idx].widthDot.data=StateVec[6]
                self.CurrentCamTracks.tracks[idx].heightDot.data=StateVec[7]

        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
            delT=(SensorData[0].header.stamp-self.CurrentRdrTracks.header.stamp) 
            delT=delT.to_sec()
            for idx in range(len(self.CurrentRdrTracks.tracks)):
                x=self.CurrentRdrTracks.tracks[idx].x.data
                y=self.CurrentRdrTracks.tracks[idx].y.data
                Vc=self.CurrentRdrTracks.tracks[idx].Vc.data
                Beta=self.CurrentRdrTracks.tracks[idx].B.data
                psi=np.array([self.psi])
                psiD=np.array([self.psiD])
                Vt=self.Vt
                F14=-delT*Vc*np.cos(((psi-Beta).astype(float)).astype(float))
                F24=delT*Vc*np.sin(((psi-Beta).astype(float)).astype(float))
                Fk=np.array([[1,delT*psiD,delT*np.sin(((psi-Beta).astype(float)).astype(float)),F14],[-delT*psiD,1,delT*np.cos(((psi-Beta).astype(float)).astype(float)),F24],[0,0,1,0],[0,0,0,1]])
                self.CurrentRdrTracks.tracks[idx].F=Mat_buildROS(Fk)
                self.CurrentRdrTracks.tracks[idx].P=Mat_buildROS(Fk*Mat_extractROS(self.CurrentRdrTracks.tracks[idx].P)*Fk.T+self.Q_rdr*(delT**2)/0.01)
                StateVec=np.array([x, y,Vc,Beta])
                A=np.array([[0,psiD,np.sin(((psi-Beta).astype(float)).astype(float)),0],[-psiD,0,0,np.cos(((psi-Beta).astype(float)).astype(float))],[0,0,0,0],[0,0,0,0]])
                StateVec=StateVec.reshape(4,1)+delT*(A.dot(StateVec.reshape(4,1))+np.array([[0],[Vt],[0],[0]]))
                self.CurrentRdrTracks.tracks[idx].x.data=StateVec[0]
                self.CurrentRdrTracks.tracks[idx].y.data=StateVec[1]
                self.CurrentRdrTracks.tracks[idx].Vc.data=StateVec[2]
                self.CurrentRdrTracks.tracks[idx].B.data=StateVec[3]


    def KalmanEstimate(self,SensorData,SensorIndices, Method):
        if isinstance(SensorData[0],CamObj): 
            Yk=self.DataAssociation(SensorData,SensorIndices,'Greedy') # The Camera always uses Greedy method
            for idx in range(len(Yk)):
                if not Yk[idx]: # No suitable measurements found, move to potential destruct
                    if  self.CurrentCamTracks.tracks[idx].Stat.data<=4:
                         self.CurrentCamTracks.tracks[idx].Stat.data+=1
                    else:
                        self.CurrentCamTracks.tracks[idx].Stat.data=4
                    continue
                else:
                    #Reset status of track as a suitable msrmt has been found
                    self.CurrentCamTracks.tracks[idx].Stat.data=1
                    track=self.CurrentCamTracks.tracks[idx]
                    StateVec=np.array([track.yPx.data,track.zPx.data,track.width.data,track.height.data,track.VyPx.data,\
                        track.VzPx.data,track.widthDot.data,track.heightDot.data]).reshape([8,1])
                    Hk=self.CamMsrtmMatrixH
                    Pk=Mat_extractROS(self.CurrentCamTracks.tracks[idx].P)
                    K=np.dot(np.dot(Pk,Hk.T),np.linalg.inv(np.dot(np.dot(Hk,Pk),Hk.T)+self.R_cam))
                    self.CurrentCamTracks.tracks[idx].K=Mat_buildROS(K)
                    YkdataAssocStateVec=np.array([(Yk[idx].xmax+Yk[idx].xmin)/2,\
                        (Yk[idx].ymax+Yk[idx].ymin)/2,\
                            (Yk[idx].xmax-Yk[idx].xmin),\
                                (Yk[idx].ymax-Yk[idx].ymin)]).reshape([4,1])
                    StateVec=StateVec+np.matmul(K,(YkdataAssocStateVec-np.matmul(Hk,StateVec)))
                    self.CurrentCamTracks.tracks[idx].yPx.data=StateVec[0]
                    self.CurrentCamTracks.tracks[idx].zPx.data=StateVec[1]
                    self.CurrentCamTracks.tracks[idx].width.data=StateVec[2]
                    self.CurrentCamTracks.tracks[idx].height.data=StateVec[3]
                    self.CurrentCamTracks.tracks[idx].VyPx.data=StateVec[4]
                    self.CurrentCamTracks.tracks[idx].VzPx.data=StateVec[5]
                    self.CurrentCamTracks.tracks[idx].widthDot.data=StateVec[6]
                    self.CurrentCamTracks.tracks[idx].heightDot.data=StateVec[7]
                    Pk=np.dot((np.eye(8)-np.dot(K,Hk)),Pk) 
                    self.CurrentCamTracks.tracks[idx].P=Mat_buildROS(Pk)
                    


        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ): # Use EKF from Truck Platooning paper:
            # DatAscTime=time.time()
            Yk=self.DataAssociation(SensorData,SensorIndices,Method) # Lists suitable measurements for each track
            # print('Time for Data Assoc:' + str(time.time()-DatAscTime))
            for idx in range(len(Yk)):
                if ((Method=='JPDA') and len(Yk[idx])==0) or ((Method=='Greedy') and (Yk[idx]==[])): # No suitable measurements found, move to potential destruct
                    if  self.CurrentRdrTracks.tracks[idx].Stat.data>=10:
                         self.CurrentRdrTracks.tracks[idx].Stat.data+=1
                    else:
                        self.CurrentRdrTracks.tracks[idx].Stat.data=10
                    continue
                else:
                    #reset Status of track:
                    self.CurrentRdrTracks.tracks[idx].Stat.data=1
                    x=np.array(self.CurrentRdrTracks.tracks[idx].x.data).astype(float)
                    y=np.array(self.CurrentRdrTracks.tracks[idx].y.data).astype(float)
                    Vc=self.CurrentRdrTracks.tracks[idx].Vc.data
                    Beta=self.CurrentRdrTracks.tracks[idx].B.data
                    psi=np.array([self.psi])
                    psiD=np.array([self.psiD])
                    Vt=self.Vt
                    posNorm=np.sqrt(x**2+y**2)
                    H31=(Vc*np.sin((psi-Beta).astype(float))*y**2-x*y*(Vc*np.cos((psi-Beta).astype(float))-Vt))/(posNorm**3)
                    H32=(-Vc*np.sin((psi-Beta).astype(float))*x*y+x**2*(Vc*np.cos((psi-Beta).astype(float))-Vt))/(posNorm**3)
                    H33=x*np.sin((psi-Beta).astype(float))/posNorm+y*np.cos((psi-Beta).astype(float))/posNorm
                    H34=(-x*Vc*np.cos((psi-Beta).astype(float))+y*Vc*np.sin((psi-Beta).astype(float)))/posNorm
                    Hk=np.array([[1,0,0,0],[x/posNorm,y/posNorm,0,0],[H31,H32,H33,H34]])
                    self.CurrentRdrTracks.tracks[idx].H=Mat_buildROS(Hk)
                    Pk=Mat_extractROS(self.CurrentRdrTracks.tracks[idx].P)
                    K=np.dot(np.dot(Pk,Hk.T),np.linalg.inv((np.dot(np.dot(Hk,Pk),Hk.T)+self.R_rdr).astype(float)))
                    self.CurrentRdrTracks.tracks[idx].K=Mat_buildROS(K)
                    StateVec=np.array([x, y,Vc,Beta]).T
                    if Method=='Greedy':
                        rho=np.sqrt(Yk[idx].pose.position.x**2+Yk[idx].pose.position.y**2)
                        rhoDot=(Yk[idx].pose.position.x*np.sin((psi-Beta).astype(float))*Vc+Yk[idx].pose.position.y*np.cos((psi-Beta).astype(float))*Vc)/rho
                        YkdataAssocStateVec=np.array([Yk[idx].pose.position.x,rho,rhoDot]).T
                        StateVec=StateVec.reshape([4,1])
                        YkdataAssocStateVec=YkdataAssocStateVec.reshape([3,1])
                        StateVec=StateVec+np.matmul(K,(YkdataAssocStateVec-np.matmul(Hk,StateVec)))
                        StateVec=StateVec.flatten()
                    else: # If using JPDA
                         StateVec=StateVec+np.matmul(K,Yk[idx])
                         StateVec=StateVec.flatten()
                    self.CurrentRdrTracks.tracks[idx].x.data=StateVec[0]
                    self.CurrentRdrTracks.tracks[idx].y.data=StateVec[1]
                    self.CurrentRdrTracks.tracks[idx].Vc.data=StateVec[2]
                    self.CurrentRdrTracks.tracks[idx].B.data=StateVec[3]
                    Pk=np.dot((np.eye(4)-np.dot(K,Hk)),Pk)
                    self.CurrentRdrTracks.tracks[idx].P=Mat_buildROS(Pk)



    def ValidationGate(self,SensorData,track):
        SensorIdxOut=[]
        if isinstance(SensorData[0],CamObj): #
            StateVec=np.array([track.yPx.data,track.zPx.data,track.width.data,track.height.data,track.VyPx.data,\
                track.VzPx.data,track.widthDot.data,track.heightDot.data])
            Hk=self.CamMsrtmMatrixH
            y_est=np.dot(Hk.reshape(4,8),StateVec.reshape(8,1))
            Pk=Mat_extractROS(track.P)
            SkInv=np.linalg.inv(np.dot(np.dot(Hk,Pk),Hk.T)+self.R_cam)
            for jdx in range(len(SensorData)):
                y=np.array([(SensorData[jdx].xmax+SensorData[jdx].xmin)/2,\
                    (SensorData[jdx].ymax+SensorData[jdx].ymin)/2,\
                        (SensorData[jdx].xmax-SensorData[jdx].xmin),\
                            (SensorData[jdx].ymax-SensorData[jdx].ymin)])
                Temp=((y.reshape(4,1)-y_est).T.dot(SkInv)).dot(y.reshape(4,1)-y_est)
                # print('GateVal')
                # print(Temp)
                # print(jdx)
                if (Temp[0]<=self.GateThreshCam**2):
                    SensorIdxOut.append(jdx)
        
        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
            StateVec=np.array([track.x.data, track.y.data,track.Vc.data,track.B.data])
            Hk=Mat_extractROS(track.H)
            psi=np.array([self.psi])
            y_est=np.dot(Hk.reshape(3,4),StateVec.reshape(4,1))
            Pk=Mat_extractROS(track.P)
            SkInv=np.linalg.inv((np.dot(np.dot(Hk,Pk),Hk.T)+self.R_rdr).astype(float))
            for jdx in range(len(SensorData)):
                Vc=np.sqrt((self.Vt+SensorData[jdx].vx)**2+SensorData[jdx].vy**2)
                if SensorData[jdx].vy==0.0:
                    Beta=0
                else:
                    Beta=SensorData[jdx].vx/SensorData[jdx].vy# This will be Vx/Vy for delphi esr
                rho=np.sqrt(SensorData[jdx].pose.position.x**2+SensorData[jdx].pose.position.y**2)
                rhoDot=(SensorData[jdx].pose.position.x*np.sin((psi-Beta).astype(float))*Vc+SensorData[jdx].pose.position.y*np.cos((psi-Beta).astype(float))*Vc)/rho
                y=np.array([SensorData[jdx].pose.position.x,rho,rhoDot])
                Temp=((y.reshape(3,1)-y_est).T.dot(SkInv)).dot(y.reshape(3,1)-y_est)
                if (Temp[0]<=self.GateThreshRdr**2):
                    SensorIdxOut.append(jdx)
        return SensorIdxOut # returns a python list, not numpy array
    def BBoxBuilder(self,data):
        self.BBoxStore=data

    def CamMsrmts(self,DataIn):
        # if DataIn.count>0:
        data=self.BBoxStore
        # data.header=DataIn.header
        self.CamReadings=[]
        for idx in range(len(data.bounding_boxes)):
            if (data.bounding_boxes[idx].id in self.YoloClassList) and (data.bounding_boxes[idx].probability>0.3): # Only add if confident of detection
            # if (data.bounding_boxes[idx].probability>0.3):
                self.CamReadings=np.append(self.CamReadings,CamObj())
                self.CamReadings[-1].header=data.header
                self.CamReadings[-1].xmin=data.bounding_boxes[idx].xmin
                self.CamReadings[-1].xmax=data.bounding_boxes[idx].xmax
                self.CamReadings[-1].ymin=data.bounding_boxes[idx].ymin
                self.CamReadings[-1].ymax=data.bounding_boxes[idx].ymax
                self.CamReadings[-1].id=data.bounding_boxes[idx].id
                self.CamReadings[-1].confidence=data.bounding_boxes[idx].probability
        self.CamReadings=np.asarray(self.CamReadings)
        #TODO: Change State Vec to just position, no widths/width rates
        self.CamRawBBPlotter(self.CamReadings)
        
        self.trackManager(self.CamReadings)


    def CamRawBBPlotter(self,SensorData):

        if not(hasattr(self,'image')) or (self.PlotArg=="1"):
            return
        LocalImage=self.image
        for idx in range(len(SensorData)):
            start=(int(SensorData[idx].xmin), int(SensorData[idx].ymin))
            end= (int(SensorData[idx].xmax),int(SensorData[idx].ymax))
            LocalImage=cv2.rectangle(LocalImage,start,end,(0,0,255),2)
        self.image=LocalImage

    def RdrMsrmtsNuSc(self,data): 
        #Build SensorData
        self.RdrReadings=[]
        
        for idx in range(len(data.objects)):
            self.RdrReadings=np.append(self.RdrReadings,RadarObj())
            self.RdrReadings[-1].pose=data.objects[idx].pose
            self.RdrReadings[-1].vx=data.objects[idx].vx
            self.RdrReadings[-1].vy=data.objects[idx].vy
            self.RdrReadings[-1].vx_comp=data.objects[idx].vx_comp
            self.RdrReadings[-1].vy_comp=data.objects[idx].vy_comp
            self.RdrReadings[-1].header=data.header
            
        self.RdrReadings=np.asarray(self.RdrReadings)
        self.trackManager(self.RdrReadings)

    def RdrMsrmtsMKZ(self,data): 
        #Build SensorData
        # startTimemst=time.time()
        self.RdrReadings=[]
        
        for idx in range(len(data.objects)):
            if (data.objects[idx].pose.pose.position.x==0.0) and (data.objects[idx].pose.pose.position.y==0.0) and (data.objects[idx].pose.covariance[0]==0.0):
                continue #Zero entry, so skip it
            self.RdrReadings=np.append(self.RdrReadings,RadarObjMKZ())
            self.RdrReadings[-1].pose=data.objects[idx].pose.pose
            self.RdrReadings[-1].vx=data.objects[idx].twist.twist.linear.x # Not used?
            self.RdrReadings[-1].vy=data.objects[idx].twist.twist.linear.y  # Not used?
            self.RdrReadings[-1].vx_comp=self.velX+data.objects[idx].twist.twist.linear.x
            self.RdrReadings[-1].vy_comp=self.velY+data.objects[idx].twist.twist.linear.y
            self.RdrReadings[-1].header=data.objects[idx].header
            self.RdrReadings[-1].id=data.objects[idx].id
            
        self.RdrReadings=np.asarray(self.RdrReadings)
        # print('Time for RdrMsrmts:' + str(time.time()-startTimemst))
        # startTimeMgr=time.time()
        self.trackManager(self.RdrReadings)
        # print('Total Time for Manager:' + str(time.time()-startTimeMgr))


if __name__=='__main__':

	main()
    

        