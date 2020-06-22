#!/usr/bin/env python3
import rospy
import numpy as np
import sys
import os
from dbw_mkz_msgs.msg import SteeringReport
from sensor_msgs.msg import Image
from derived_object_msgs.msg import ObjectWithCovarianceArray
# from delphi_esr_msgs.msg import EsrEthTx
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from darknet_ros_msgs.msg import BoundingBoxes
from nuscenes2bag.msg import RadarObjects
from utils import Mat_buildROS
from utils import Mat_extractROS
from flir_adk_multi.msg import trackArrayRdr
from flir_adk_multi.msg import trackRdr
from flir_adk_multi.msg import trackArrayCam
from flir_adk_multi.msg import trackCam

from utils import CamObj
from utils import RadarObj
from utils import RadarObjMKZ
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError
import cv2
from itertools import permutations
def main():
    rospy.init_node('jpda', anonymous=True)
    DataSetType=sys.argv[1]
    Method=sys.argv[2]
    # print(inputArg)
    fusInst=jpda_class(DataSetType,Method)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
class jpda_class():

    def __init__(self,DataSetType,Method):
        self.TrackPubRdr=rospy.Publisher("dataAssocRdr",trackArrayRdr, queue_size=1000) 
        self.TrackPubCam=rospy.Publisher("dataAssocCam",trackArrayCam, queue_size=1000) 
        self.image_pub=rospy.Publisher("fusedImage",Image, queue_size=1000) 
        self.YoloClassList=[0,1,2,3,5,7] # For NuSc
        self.YoloClassList=[0,1,2] # For Yolov3_flir
        self.GateThreshRdr =1# Scaling factor, threshold for gating
        self.GateThreshCam=3# TODO: adjust?
        self.trackInitRdrThresh=0.2 # For track initiation
        self.trackInitCamThresh=15 # Radius of 15 pixels allowed
        self.bridge=CvBridge()
        self.font=cv2.FONT_HERSHEY_SIMPLEX 
        self.image=[] # Initialized
        # Initializing parameters:
        self.Q_rdr=np.array([[0.1,0,0,0],[0,0.11,0,0],[0,0,0.1,0],[0,0,0,0.1]])
        self.R_rdr=np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.Q_cam=0.1*np.eye(8)
        self.R_cam=np.array([[3,0,0,0],[0,3,0,0],[0,0,15,0],[0,0,0,15]])
        self.CamMsrtmMatrixH=np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],\
            [0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]] # Only positions and w/h are measured)
        self.Vt=[]
        self.velY=[]
        self.velX=[]
        self.psi=[]
        self.psiD=[]# psiDot
        self.Method=Method
        self.CamXOffset=2.36#=93 inches, measured b/w cam and Rdr, in x direction
        self.CamZoffset=1 # Roughly 40 inches
        if DataSetType=="NuSc":
            rospy.Subscriber('/cam_front/raw', Image, self.buildImage)
            rospy.Subscriber('/vel', Twist, self.Odom1NuSc) 
            rospy.Subscriber('/odom', Odometry, self.Odom2NuSc) 
            rospy.Subscriber('/imu', Imu, self.Odom3NuSc)
            rospy.Subscriber('/radar_front', BoundingBoxes, self.RdrMsrmtsNuSc)
        elif DataSetType=="MKZ":
            rospy.Subscriber('/Thermal_Panorama', Image, self.buildImage)
            rospy.Subscriber('/vehicle/steering_report', SteeringReport, self.Odom1MKZ) 
            rospy.Subscriber('/vehicle/gps/vel', TwistStamped, self.Odom2MKZ) # TODO: fix after IMU is available
            rospy.Subscriber('/vehicle/twist', TwistStamped,self.Odom3MKZ)
            rospy.Subscriber('/as_tx/objects', ObjectWithCovarianceArray, self.RdrMsrmtsMKZ)
            rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.CamMsrmts)
        # rospy.Subscriber('/as_tx/objects')
        rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.CamMsrmts)
        # rospy.Subscriber('/radar_front', BoundingBoxes, self.RdrMsrmtsMKZ)
        #TODO: if-else switches for trackInitiator, trackDestructor, and Kalman functions for radar and camera
        
    def buildImage(self,data):
        # print('gotImage')
        self.image=self.bridge.imgmsg_to_cv2(data, "rgb8")

    def Odom1NuSc(self,data):
        self.Vt =data.linear.x 
    def Odom2NuSc(self,data):
        self.psi=data.pose.pose.orientation.z
    def Odom3NuSc(self,data):
        self.psiD=data.angular_velocity.z
    def Odom1MKZ(self,data):
        self.Vt=data.speed
    def Odom2MKZ(self,data):
        self.psi=np.arctan2(data.twist.linear.y,data.twist.linear.x)
        self.velX=data.twist.linear.x
        self.velY=data.twist.linear.y
        # print(self.psi)
    def Odom3MKZ(self,data):
        self.psiD=data.twist.angular.z





    def trackInitiator(self,SensorData):
        if not any(SensorData):
            return
        elif isinstance(SensorData[0],CamObj): 
            if hasattr(self, 'InitiatedCamTracks'):
                # Then, move to current tracks based on NN-style gating
                toDel=[]
                for idx in range(len(self.InitiatedCamTracks.tracks)):
                    for jdx in range(len(SensorData)):
                        R=np.sqrt((self.InitiatedCamTracks.tracks[idx].yPx-(SensorData[jdx].xmax+SensorData[jdx].xmin)/2)^2 \
                            +(self.InitiatedCamTracks.tracks[idx].zPx-(SensorData[jdx].ymax+SensorData[jdx].ymin)/2)^2)
                            #TODO: The initiation doesnt account for those SensorData that are already being tracked by CurrentTracks
                        if R<=self.trackInitCamThresh: # Then move this to current track # Inherent assumption here is that only one will be suitable
                            if  not hasattr(self, 'CurrentCamTracks'):
                                self.CurrentCamTracks=trackArrayCam()
                            delT=SensorData[0].header-self.InitiatedCamTracks.header
                            self.CurrentCamTracks.header=SensorData[jdx].header
                            self.InitiatedCamTracks.tracks[idx].Stat.data=1 # Moving to current track
                            # Update the track with new sensor data before pushing to Current tracks
                            self.InitiatedCamTracks.tracks[idx].VyPx.data=\
                                (self.InitiatedCamTracks.tracks[idx].yPx-SensorData[jdx].(SensorData[jdx].xmax+SensorData[jdx].xmin)/2)/delT
                            self.InitiatedCamTracks.tracks[idx].VzPy.data=\
                                (self.InitiatedCamTracks.tracks[idx].zPx-SensorData[jdx].(SensorData[jdx].ymax+SensorData[jdx].ymin)/2)/delT
                            self.InitiatedCamTracks.tracks[idx].widthDot.data=\
                                (self.InitiatedCamTracks.tracks[idx].width.data-(SensorData[jdx].xmax-SensorData[jdx].xmin))/delT
                            self.InitiatedCamTracks.tracks[idx].heightDot.data=\
                                (self.InitiatedCamTracks.tracks[idx].height.data-(SensorData[jdx].ymax-SensorData[jdx].ymin))/delT
                            self.InitiatedCamTracks.tracks[idx].height.data=(SensorData[jdx].ymax-SensorData[jdx].ymin)
                            self.InitiatedCamTracks.tracks[idx].width.data=(SensorData[jdx].xmax-SensorData[jdx].xmin)
                            self.InitiatedCamTracks.tracks[idx].yPx.data=(SensorData[jdx].xmax+SensorData[jdx].xmin)/2
                            self.InitiatedCamTracks.tracks[idx].zPx.data=(SensorData[jdx].ymax+SensorData[jdx].ymin)/2
                            Pk=np.diag([3,3,10,10,3,3,3,3])
                            self.InitiatedRdrTracks.tracks[idx].P=Mat_buildROS(Pk)
                            self.CurrentCamTracks.tracks=np.append(self.CurrentCamTracks.tracks,self.InitiatedCamTracks.tracks[idx])
                            toDel.append(idx)
                            SensorData=np.delete(SensorData,jdx)
                            break
                        elif jdx==(len(SensorData)-1): # for this idx of InitiatedCamTrack, the last jdx, so no measurements are nearby; delete the idx
                            toDel.append(idx)
                    # Clean all InitiatedCamTracks using toDel
                    self.InitiatedCamTracks.tracks=np.delete(self.InitiatedCamTracks.tracks,toDel)
                    #Remove old initiated tracks (if idle for more than 3 time steps):
                    toDel2=[]
                    for idx in range(len(self.InitiatedCamTracks.tracks)):
                        self.InitiatedCamTracks.tracks[idx].Stat.data=self.InitiatedCamTracks.tracks[idx].Stat.data-1
                        if self.InitiatedCamTracks.tracks[idx].Stat.data<=-4
                        toDel2.append(idx)
                    self.InitiatedCamTracks.tracks=np.delete(self.InitiatedCamTracks.tracks,toDel2)
                    # Then concatenate remaining sensor Data for future initation
                    self.InitiatedCamTracks.header=SensorData[0].header
                    for idx in range(len(SensorData)):
                        self.InitiatedCamTracks.tracks=np.append(self.InitiatedCamTracks.tracks,trackCam())
                        self.InitiatedCamTracks.tracks[-1].Stat.data= -1 # InitiatedTrack
                        self.InitiatedCamTracks.tracks[-1].yPx.data=(SensorData[idx].xmax+SensorData[idx].xmin)/2
                        self.InitiatedCamTracks.tracks[-1].zPx.data=(SensorData[idx].ymax+SensorData[idx].ymin)/2
                        self.InitiatedCamTracks.tracks[-1].VyPx.data=0
                        self.InitiatedCamTracks.tracks[-1].VzPy.data=0
                        self.InitiatedCamTracks.tracks[-1].width.data=(SensorData[idx].xmax-SensorData[idx].xmin)
                        self.InitiatedCamTracks.tracks[-1].widthDot.data=0
                        self.InitiatedCamTracks.tracks[-1].height.data=(SensorData[idx].ymax-SensorData[idx].ymin)
                        self.InitiatedCamTracks.tracks[-1].heightDot.data=0

            else: # Start of algorithm, no tracks
                self.InitiatedCamTracks=trackArrayCam()
                self.InitiatedCamTracks.header=SensorData[0].header
                for idx in range(len(SensorData)):
                    self.InitiatedCamTracks.tracks=np.append(self.InitiatedCamTracks.tracks,trackCam())
                    self.InitiatedCamTracks.tracks[-1].Stat.data=-1 # Initiated Track
                    self.InitiatedCamTracks.tracks[-1].yPx.data=(SensorData[idx].xmax+SensorData[idx].xmin)/2
                    self.InitiatedCamTracks.tracks[-1].zPx.data=(SensorData[idx].ymax+SensorData[idx].ymin)/2
                    self.InitiatedCamTracks.tracks[-1].VyPx.data=0
                    self.InitiatedCamTracks.tracks[-1].VzPy.data=0
                    self.InitiatedCamTracks.tracks[-1].width.data=(SensorData[idx].xmax-SensorData[idx].xmin)
                    self.InitiatedCamTracks.tracks[-1].widthDot.data=0
                    self.InitiatedCamTracks.tracks[-1].height.data=(SensorData[idx].ymax-SensorData[idx].ymin)
                    self.InitiatedCamTracks.tracks[-1].heightDot.data=0
                





            
        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
            
            if hasattr(self, 'InitiatedRdrTracks'):# Some (or Zer0) tracks already exists (i.e, not start of algorithm)
                
                toDel=[]

                for idx in range(len(self.InitiatedRdrTracks.tracks)):
                    gateValX=[]
                    gateValY=[]
                    gateValRMS=[]
                    # Find all sensor objects within some gate
                    for jdx in range(len(SensorData)):
                        gateValX.append(np.abs(SensorData[jdx].pose.position.x-self.InitiatedRdrTracks.tracks[idx].x.data))
                        gateValY.append(np.abs(SensorData[jdx].pose.position.y-self.InitiatedRdrTracks.tracks[idx].y.data))
                        gateValRMS.append(np.sqrt((gateValX[jdx])**2+(gateValY[jdx])**2))
                    if (np.min(np.array(gateValRMS))<=self.trackInitRdrThresh): # @50Hz, 20m/s in X dir and 10m/s in Y-Direction as validation gate
                        #If gate is satisfied, move to CurrentRdrTracks after initiating P and deleting that SensorData[idx]
                        self.InitiatedRdrTracks.tracks[idx].P=Mat_buildROS(np.array([[1,0,0,0],[0,1,0,0],[0,0,0.5,0],[0,0,0,2]]))
                        #(Large uncertainity given to Beta. Others conservatively picked based on Delphi ESR spec sheet)
                        self.InitiatedRdrTracks.tracks[idx].Stat.data=1# Moving to CurrentRdrTracks
                        x=self.InitiatedRdrTracks.tracks[idx].x.data
                        y=self.InitiatedRdrTracks.tracks[idx].y.data
                        Vc=self.InitiatedRdrTracks.tracks[idx].Vc.data
                        Beta=self.InitiatedRdrTracks.tracks[idx].B.data
                        psi=self.psi
                        psiD=self.psiD
                        Vt=self.Vt
                        posNorm=np.sqrt(x**2+y**2)
                        H31=(Vc*np.sin(psi-Beta)*y**2-x*y*(Vc*np.cos(psi-Beta)-Vt))/(posNorm**3)
                        H32=(-Vc*np.sin(psi-Beta)*x*y+x**2*(Vc*np.cos(psi-Beta)-Vt))/(posNorm**3)
                        H33=x*np.sin(psi-Beta)/posNorm+y*np.cos(psi-Beta)/posNorm
                        H34=(-x*Vc*np.cos(psi-Beta)+y*Vc*np.sin(psi-Beta))/posNorm
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
                    if self.InitiatedRdrTracks.tracks[idx].Stat.data<=-3
                    toDel2.append(idx)
                self.InitiatedRdrTracks.tracks=np.delete(self.InitiatedRdrTracks.tracks,toDel2)
                # Then concatenate remaining sensor Data for future initation
                self.InitiatedRdrTracks.header=SensorData[0].header
                for idx in range(len(SensorData)):
                    self.InitiatedRdrTracks.tracks=np.append(self.InitiatedRdrTracks.tracks,trackRdr())
                    self.InitiatedRdrTracks.tracks[-1].Stat.data= -1 # InitiatedTrack
                    self.InitiatedRdrTracks.tracks[-1].x.data=SensorData[idx].pose.position.x
                    self.InitiatedRdrTracks.tracks[-1].y.data=SensorData[idx].pose.position.y
                    self.InitiatedRdrTracks.tracks[-1].Vc.data=np.sqrt(SensorData[idx].vx_comp**2+SensorData[idx].vy_comp**2)
                    self.InitiatedRdrTracks.tracks[-1].B.data=self.psi # TODO: Have better estimate for Beta during intialization

            else: # Start of algorithm, no tracks
                
                # print('createdCurrentRdrTracks')
                self.InitiatedRdrTracks=trackArrayRdr()
                self.InitiatedRdrTracks.header=SensorData[0].header
                for idx in range(len(SensorData)):
                    self.InitiatedRdrTracks.tracks=np.append(self.InitiatedRdrTracks.tracks,trackRdr())
                    self.InitiatedRdrTracks.tracks[-1].Stat.data= -1 # InitiatedTrack
                    self.InitiatedRdrTracks.tracks[-1].x.data=SensorData[idx].pose.position.x
                    self.InitiatedRdrTracks.tracks[-1].y.data=SensorData[idx].pose.position.y
                    self.InitiatedRdrTracks.tracks[-1].Vc.data=np.sqrt(SensorData[idx].vx_comp**2+SensorData[idx].vy_comp**2)
                    self.InitiatedRdrTracks.tracks[-1].B.data=self.psi # TODO: Have better estimate for Beta during intialization
                

    def trackDestructor(self,SensorData):
        if isinstance(SensorData[0],CamObj): 
            toDel=[]
            for idx in range(len(self.CurrentCamTracks.tracks)):
                if self.CurrentCamTracks.tracks[idx].Stat.data>=15:# If no measurements associated for 5 steps
                    toDel.append(idx)
            self.CurrentCamTracks.tracks=np.delete(self.CurrentCamTracks.tracks,toDel)
        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
            toDel=[]
            for idx in range(len(self.CurrentRdrTracks.tracks)):
                if self.CurrentRdrTracks.tracks[idx].Stat.data>=15: # If no measurements associated for 5 steps
                    toDel.append(idx)
            self.CurrentRdrTracks.tracks=np.delete(self.CurrentRdrTracks.tracks,toDel)

    def trackMaintenance(self,SensorData):
        if not (hasattr(self, 'CurrentRdrTracks') and hasattr(self, 'CurrentCamTracks')):
            return
        if isinstance(SensorData[0],CamObj): 
            SensorIndices=[]
            for idx in range(len(self.CurrentCamTracks.tracks)):
                SensorIndices.append(self.ValidationGate(SensorData,self.CurrentCamTracks.tracks[idx]))#Clean the incoming data - outputs 2D python array
                # Above yields array of possible measurments (only indices) corresponding to a particular track
            self.KalmanEstimate(SensorData,SensorIndices, self.Method) # Includes DataAssociation Calcs
            self.KalmanPropagate(SensorData)
            self.TrackPubCam.publish(tracks =self.CurrentCamTracks.tracks)
        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
            SensorIndices=[]
            for idx in range(len(self.CurrentRdrTracks.tracks)):
                SensorIndices.append(self.ValidationGate(SensorData,self.CurrentRdrTracks.tracks[idx]))#Clean the incoming data - outputs 2D python array
                # Above yields array of possible measurments (only indices) corresponding to a particular track
            self.KalmanEstimate(SensorData,SensorIndices, self.Method) # Includes DataAssociation Calcs
            self.KalmanPropagate(SensorData)
            self.TrackPubRdr.publish(tracks =self.CurrentRdrTracks.tracks)
            # rospy.loginfo('Current tracks published to topic /dataAssoc')
            

    def trackPlotter(self):
        if len(self.image)==0:
           return # Skip function call if image is not available
        n=len(self.CurrentRdrTracks.tracks)
        m=len(self.CurrentCamTracks.tracks)
        RadarAnglesH=np.zeros((n,1))
        RadarAnglesV=np.zeros((n,1))
        # Camera Coordinates: X is horizontal, Y is vertical starting from left top corner
        CirClr=[]
        LocalImage=self.image
        for idx in range(n):
            RadarAnglesH[idx]=-np.degrees(np.arctan(np.divide(self.CurrentRdrTracks.tracks[idx].y.data,self.CurrentRdrTracks.tracks[idx].x.data)))
            RadarAnglesV[idx]=np.abs(np.degrees(np.arctan(np.divide(self.CamZoffset,self.CurrentRdrTracks.tracks[idx].x.data+self.CamXOffset)))) #will always be negative, so correct for it
            if (self.CurrentRdrTracks.tracks[idx].Stat.data>=1) and (self.CurrentRdrTracks.tracks[idx].Stat.data<14): #Current Track- green
                CirClr.append(np.array([0,255,0]))
            elif self.CurrentRdrTracks.tracks[idx].Stat.data<=0: # Candidate Tracks for initialization - blue
                CirClr.append(np.array([255,0,0]))
            else: # Candidate for Destructor-orange
                CirClr.append(np.array([0,165,255])) 
        CameraX=RadarAnglesH*(self.image.shape[1]/190) + self.image.shape[1]/2 # Number of pixels per degree,adjusted for shifting origin from centerline to top left
        CameraY=RadarAnglesV*(self.image.shape[0]/39.375) +512/2 # Number of pixels per degree,adjusted for shifting origin from centerline to top left
        CirClr=np.array(CirClr)
        for idx in range(len(RadarAnglesH)):
            if (CameraX[idx]<=self.image.shape[1]):
                LocalImage=cv2.circle(LocalImage, (int(CameraX[idx]),int(CameraY[idx])), 12, CirClr[idx].tolist(),3)
                LocalImage=cv2.putText(LocalImage,str(idx),(int(CameraX[idx]),int(CameraY[idx])),self.font,1,(255,105,180),2)
        
        # Now set colors for the Camera Tracks and plot them:
        RectClr=[]
        for jdx in range(m):
            if (self.CurrentCamTracks.tracks[idx].Stat.data>=1) and (self.CurrentCamTracks.tracks[idx].Stat.data<14): #Current Track- green
                RectClr.append(np.array([0,255,0]))
            elif self.CurrentCamTracks.tracks[idx].Stat.data<=0: # Candidate Tracks for initialization - blue
                RectClr.append(np.array([255,0,0]))
            else: # Candidate for Destructor-orange
                RectClr.append(np.array([0,165,255])) 
        for idx in range(len(RectClr)):
            LocalImage=cv2.rectangle(LocalImage,\
                (self.CurrentCamTracks.tracks[idx].yPx-self.CurrentCamTracks.tracks[idx].widht/2,self.CurrentCamTracks.tracks[idx].zPx-self.CurrentCamTracks.tracks[idx].height/2),\
                    (self.CurrentCamTracks.tracks[idx].yPx+self.CurrentCamTracks.tracks[idx].widht/2,self.CurrentCamTracks.tracks[idx].zPx+self.CurrentCamTracks.tracks[idx].height/2),\
                         RectClr[idx].tolist(),2)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(LocalImage, "bgr8"))
        rospy.loginfo('Image is being published')
   
    def CamRdrCombine(self):
        pass

    def trackManager(self,SensorData):
        self.trackInitiator(SensorData)
        self.trackMaintenance(SensorData)
        self.trackDestructor(SensorData)
        self.trackPlotter()
        self.CamRdrCombine()
        print('No. of Radar Tracks: ',end ="")
        print(len(self.CurrentRdrTracks.tracks))
        print('No. of Camera Tracks: ',end ="")
        print(len(self.CurrentRdrTracks.tracks))
        # TODO: combine radar and cam track info - REMEMBER THAT yPX ,zPX etc is in coordinate frame, need to translate and change sign etc...
        

                
    def DataAssociation(self,SensorData,SensorIndices,Method):
        if Method=="Hungarian":
            pass
        elif Method=="JPDA":
            #Build A Validation Matrix if there are sufficient sensor data and tracks
            if (len(SensorData)<1) or (len(self.CurrentRdrTracks.tracks)<1):
                Yk=[]
            else:
                # ValidationMat=np.zeros((len(SensorData),len(self.CurrentRdrTracks.tracks)+1),dtype=float)
                # ValidationMat[:,0]=np.ones((len(SensorData),1))[:,0]
                # # for idx in range(len(SensorData)): #Rows
                # #     for jdx in range(len(SensorIndices[idx])):
                # #         ValidationMat[idx][jdx+1]=1
                # # Now select different permutations:
                # seedList=np.append(np.zeros((1,len(SensorData)-1)),1)
                # l=list(permutations(seedList.tolist())) 
                # L=np.array(l).T
                # print(L.shape)
                Phi=len(SensorData)-len(self.CurrentRdrTracks.tracks) # Number of False measurements
                C=3 # Number of false measurements per unit volume (assume)
                Pd=0.9 #Probability of detection
                # Find Nt(j) - the number of targets asociated with each observation j
                Nt=np.empty([len(SensorData),1])
                print(Nt)
                for jdx in range(len(SensorData)):
                    Nt[jdx]=SensorIndices.count(jdx)
                print(Nt)
                if (Nt==0).all():
                    Yk=[]
                else:
                    pass




            return Yk


                
        elif Method=="Greedy": # Simple method that just outputs the closest UNUSED measurement
            if isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
                # Sensor indices is a 2D python list, not numpy array
                usedSensorIndices=[]
                Yk=[] # A python list of sensor measurements corresponding to each CurrentTrack
                for idx in range(len(self.CurrentRdrTracks.tracks)):
                    # print('dataAssoc Yk idx:')
                    # print(idx)
                    gateValX=[]
                    gateValY=[]
                    gateValRMS=[]
                    if len(SensorIndices[idx])==0:
                        Yk.append([])
                        continue
                    else:
                        for jdx in range(len(SensorIndices[idx])):
                            gateValX.append(np.abs(SensorData[SensorIndices[idx][jdx]].pose.position.x-self.CurrentRdrTracks.tracks[idx].x.data))
                            gateValY.append(np.abs(SensorData[SensorIndices[idx][jdx]].pose.position.y-self.CurrentRdrTracks.tracks[idx].y.data))
                            gateValRMS.append(np.sqrt((gateValX[jdx])**2+(gateValY[jdx])**2))
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
            elif isinstance(SensorData[0],CamObj): # Just give the first measurement (Again, assuming that there will only be one close measrument)
                Yk=[]
                for idx in range(len(self.CurrentCamTracks.tracks)):
                    if len(SensorIndices[idx])==0:
                        Yk.append([])
                        continue
                    else:
                        Yk.append(SensorData[SensorIndices[idx][0]])

        return Yk # An Array with same len as CurrentRdrTracks.tracks[]

    def KalmanPropagate(self,SensorData):
        delT=(SensorData[0].header.stamp-self.CurrentRdrTracks.header.stamp) 
        delT=delT.to_sec()
        if isinstance(SensorData[0],CamObj):
            for idx in range(len(self.CurrentCamTracks.tracks)):
                Fk=np.eye(8)
                Fk[0,4]=delT
                Fk[1,5]=delT
                Fk[2,6]=delT
                Fk[3,7]=delT
                self.CurrentCamTracks.tracks[idx].P=Mat_buildROS(Fk*Mat_extractROS(self.CurrentCamTracks.tracks[idx].P)*Fk.T+self.Q_cam)
                track=self.CurrentCamTracks.tracks[idx]
                StateVec=np.array([track.yPx.data,track.zPx.data,track.width.data,track.height.data,track.VyPx.data,\
                        track.VzPx.data,track.widhtDot.data,track.heightDot.data])
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
            for idx in range(len(self.CurrentRdrTracks.tracks)):
                x=self.CurrentRdrTracks.tracks[idx].x.data
                y=self.CurrentRdrTracks.tracks[idx].y.data
                Vc=self.CurrentRdrTracks.tracks[idx].Vc.data
                Beta=self.CurrentRdrTracks.tracks[idx].B.data
                psi=self.psi
                psiD=self.psiD
                Vt=self.Vt
                F14=-delT*Vc*np.cos(psi-Beta)
                F24=delT*Vc*np.sin(psi-Beta)
                Fk=np.array([[1,delT*psiD,delT*np.sin(psi-Beta),F14],[-delT*psiD,1,delT*np.cos(psi-Beta),F24],[0,0,1,0],[0,0,0,1]])
                self.CurrentRdrTracks.tracks[idx].F=Mat_buildROS(Fk)
                self.CurrentRdrTracks.tracks[idx].P=Mat_buildROS(Fk*Mat_extractROS(self.CurrentRdrTracks.tracks[idx].P)*Fk.T+self.Q_rdr*(delT**2)/0.01)
                StateVec=np.array([x, y,Vc,Beta])
                A=np.array([[0,psiD,np.sin(psi-Beta),0],[-psiD,0,0,np.cos(psi-Beta)],[0,0,0,0],[0,0,0,0]])
                StateVec=StateVec.reshape(4,1)+delT*(A.dot(StateVec.reshape(4,1))+np.array([[0],[Vt],[0],[0]]))
                self.CurrentRdrTracks.tracks[idx].x.data=StateVec[0]
                self.CurrentRdrTracks.tracks[idx].y.data=StateVec[1]
                self.CurrentRdrTracks.tracks[idx].Vc.data=StateVec[2]
                self.CurrentRdrTracks.tracks[idx].B.data=StateVec[3]


    def KalmanEstimate(self,SensorData,SensorIndices, Method):
        # print('reached KEST')
        if isinstance(SensorData[0],CamObj): 
            Yk=self.DataAssociation(SensorData,SensorIndices,Method)
            for idx in range(len(Yk)):
                if not Yk[idx]: # No suitable measurements found, move to potential destruct
                    if  self.CurrentCamTracks.tracks[idx].Stat.data>=10:
                         self.CurrentCamTracks.tracks[idx].Stat.data+=1
                    else:
                        self.CurrentCamTracks.tracks[idx].Stat.data=10
                    continue
                else:
                    #Reset status of track as a suitable msrmt has been found
                    self.CurrentCamTracks.tracks[idx].Stat.data=1
                    StateVec=np.array([track.yPx.data,track.zPx.data,track.width.data,track.height.data,track.VyPx.data,\
                        track.VzPx.data,track.widhtDot.data,track.heightDot.data])
                    Hk=self.CamMsrtmMatrixH
                    Pk=Mat_extractROS(self.CurrentCamTracks.tracks[idx].P)
                    K=np.dot(np.dot(Pk,Hk.T),np.linalg.inv(np.dot(np.dot(Hk,Pk),Hk.T)+self.R_cam))
                    self.CurrentRdrTracks.tracks[idx].K=Mat_buildROS(K)
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
                    Pk=np.dot((np.eye(4)-np.dot(K,Hk)),Pk) # TODO: correct this size 
                    self.CurrentCamTracks.tracks[idx].P=Mat_buildROS(Pk)
                    


        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ): # Use EKF from Truck Platooning paper:
            Yk=self.DataAssociation(SensorData,SensorIndices,Method) # Lists suitable measurements for each track
            for idx in range(len(Yk)):
                if not Yk[idx]: # No suitable measurements found, move to potential destruct
                    if  self.CurrentRdrTracks.tracks[idx].Stat.data>=10:
                         self.CurrentRdrTracks.tracks[idx].Stat.data+=1
                    else:
                        self.CurrentRdrTracks.tracks[idx].Stat.data=10
                    continue
                else:
                    #reset Status of track:
                    self.CurrentRdrTracks.tracks[idx].Stat.data=1
                    x=self.CurrentRdrTracks.tracks[idx].x.data
                    y=self.CurrentRdrTracks.tracks[idx].y.data
                    Vc=self.CurrentRdrTracks.tracks[idx].Vc.data
                    Beta=self.CurrentRdrTracks.tracks[idx].B.data
                    psi=self.psi
                    psiD=self.psiD
                    Vt=self.Vt
                    posNorm=np.sqrt(x**2+y**2)
                    H31=(Vc*np.sin(psi-Beta)*y**2-x*y*(Vc*np.cos(psi-Beta)-Vt))/(posNorm**3)
                    H32=(-Vc*np.sin(psi-Beta)*x*y+x**2*(Vc*np.cos(psi-Beta)-Vt))/(posNorm**3)
                    H33=x*np.sin(psi-Beta)/posNorm+y*np.cos(psi-Beta)/posNorm
                    H34=(-x*Vc*np.cos(psi-Beta)+y*Vc*np.sin(psi-Beta))/posNorm
                    Hk=np.array([[1,0,0,0],[x/posNorm,y/posNorm,0,0],[H31,H32,H33,H34]])
                    self.CurrentRdrTracks.tracks[idx].H=Mat_buildROS(Hk)
                    Pk=Mat_extractROS(self.CurrentRdrTracks.tracks[idx].P)
                    K=np.dot(np.dot(Pk,Hk.T),np.linalg.inv(np.dot(np.dot(Hk,Pk),Hk.T)+self.R_rdr))
                    self.CurrentRdrTracks.tracks[idx].K=Mat_buildROS(K)
                    StateVec=np.array([x, y,Vc,Beta]).T
                    # print(Yk[idx])
                    rho=np.sqrt(Yk[idx].pose.position.x**2+Yk[idx].pose.position.y**2)
                    rhoDot=(Yk[idx].pose.position.x*np.sin(self.psi-Beta)*Vc+Yk[idx].pose.position.y*np.cos(self.psi-Beta)*Vc)/rho
                    YkdataAssocStateVec=np.array([Yk[idx].pose.position.x,rho,rhoDot]).T
                    StateVec=StateVec.reshape([4,1])
                    YkdataAssocStateVec=YkdataAssocStateVec.reshape([3,1])
                    # print(np.matmul(Hk,StateVec).shape)
                    # print(YkdataAssocStateVec.shape)
                    # print(K.shape)
                    # print(YkdataAssocStateVec.shape)
                    StateVec=StateVec+np.matmul(K,(YkdataAssocStateVec-np.matmul(Hk,StateVec)))
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
                track.VzPx.data,track.widhtDot.data,track.heightDot.data])
            Hk=self.CamMsrtmMatrixH
            # print(StateVec.reshape(4,1)[1])
            y_est=np.dot(Hk.reshape(4,8),StateVec.reshape(8,1))
            Pk=Mat_extractROS(track.P)
            SkInv=np.linalg.inv(np.dot(np.dot(Hk,Pk),Hk.T)+self.R_cam)
            for jdx in range(len(SensorData)):
                y=np.array([(SensorData[jdx].xmax+SensorData[jdx].xmin)/2,\
                    (SensorData[jdx].ymax+SensorData[jdx].ymin)/2,\
                        (SensorData[jdx].xmax-SensorData[jdx].xmin),\
                            (SensorData[jdx].ymax-SensorData[jdx].ymin)])
                Temp=((y.reshape(4,1)-y_est).T.dot(SkInv)).dot(y.reshape(4,1)-y_est)
                if (Temp[0]<=self.GateThreshCam**2):
                    SensorIdxOut.append(jdx)
        
        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
            StateVec=np.array([track.x.data, track.y.data,track.Vc.data,track.B.data])
            Hk=Mat_extractROS(track.H)
            # print(StateVec.reshape(4,1)[1])
            y_est=np.dot(Hk.reshape(3,4),StateVec.reshape(4,1))
            Pk=Mat_extractROS(track.P)
            # print('SkInv:')
            # print(np.dot(np.dot(Hk,Pk),Hk.T)+self.R_rdr)
            SkInv=np.linalg.inv(np.dot(np.dot(Hk,Pk),Hk.T)+self.R_rdr)
            for jdx in range(len(SensorData)):
                Vc=np.sqrt((self.Vt+SensorData[jdx].vx)**2+SensorData[jdx].vy**2)
                if SensorData[jdx].vy==0.0:
                    Beta=0
                else:
                    Beta=  SensorData[jdx].vx/SensorData[jdx].vy# This will be Vx/Vy for delphi esr
                rho=np.sqrt(SensorData[jdx].pose.position.x**2+SensorData[jdx].pose.position.y**2)
                rhoDot=(SensorData[jdx].pose.position.x*np.sin(self.psi-Beta)*Vc+SensorData[jdx].pose.position.y*np.cos(self.psi-Beta)*Vc)/rho
                y=np.array([SensorData[jdx].pose.position.x,rho,rhoDot])
                Temp=((y.reshape(3,1)-y_est).T.dot(SkInv)).dot(y.reshape(3,1)-y_est)
                if (Temp[0]<=self.GateThreshRDR**2):
                    SensorIdxOut.append(jdx)
        return SensorIdxOut # returns a python list, not numpy array

    def CamMsrmts(self,data):
        self.CamReadings=[]
        for idx in range(len(data.bounding_boxes)):
            #print(data.bounding_boxes[2].id)
            if data.bounding_boxes[idx].id in self.YoloClassList:
                self.CamReadings=np.append(self.CamReadings,CamObj())
                self.CamReadings[-1].header=data.header
                self.CamReadings[-1].xmin=data.bounding_boxes[idx].xmin
                self.CamReadings[-1].xmax=data.bounding_boxes[idx].xmax
                self.CamReadings[-1].ymin=data.bounding_boxes[idx].ymin
                self.CamReadings[-1].ymax=data.bounding_boxes[idx].ymax
                self.CamReadings[-1].id=data.bounding_boxes[idx].id
                self.CamReadings[-1].confidence=data.bounding_boxes[idx].probability
        self.CamReadings=np.asarray(self.CamReadings)
        self.trackManager(self.CamReadings)

    def RdrMsrmtsNuSc(self,data): 
        #Build SensorData
        # print('RdrCallback')
        self.RdrReadings=[]
        
        for idx in range(len(data.objects)):
            self.RdrReadings=np.append(self.RdrReadings,RadarObj())
            self.RdrReadings[-1].pose=data.objects[idx].pose
            self.RdrReadings[-1].vx=data.objects[idx].vx
            self.RdrReadings[-1].vy=data.objects[idx].vy
            self.RdrReadings[-1].vx_comp=data.objects[idx].vx_comp
            self.RdrReadings[-1].vy_comp=data.objects[idx].vy_comp
            self.RdrReadings[-1].header=data.header
            
        #print(self.RdrReadings[3].stamp)
        self.RdrReadings=np.asarray(self.RdrReadings)
        self.trackManager(self.RdrReadings)

    def RdrMsrmtsMKZ(self,data): 
        #Build SensorData
        # print('RdrCallback')
        self.RdrReadings=[]
        
        for idx in range(len(data.objects)):
            self.RdrReadings=np.append(self.RdrReadings,RadarObjMKZ())
            self.RdrReadings[-1].pose=data.objects[idx].pose.pose
            self.RdrReadings[-1].vx=data.objects[idx].twist.twist.linear.x
            self.RdrReadings[-1].vy=data.objects[idx].twist.twist.linear.y
            self.RdrReadings[-1].vx_comp=self.velX+data.objects[idx].twist.twist.linear.x
            self.RdrReadings[-1].vy_comp=self.velY+data.objects[idx].twist.twist.linear.y
            self.RdrReadings[-1].header=data.objects[idx].header
            self.RdrReadings[-1].id=data.objects[idx].id
            
        #print(self.RdrReadings[3].stamp)
        self.RdrReadings=np.asarray(self.RdrReadings)
        self.trackManager(self.RdrReadings)


if __name__=='__main__':

	main()
    

        