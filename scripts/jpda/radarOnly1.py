#!/usr/bin/env python3
import rospy
import numpy as np
import sys
import os
# import dbw_mkz_msgs.msg
from sensor_msgs.msg import Image
# from delphi_esr_msgs.msg import EsrEthTx
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from darknet_ros_msgs.msg import BoundingBoxes
from nuscenes2bag.msg import RadarObjects
from utils import Mat_buildROS
from utils import Mat_extractROS
from flir_adk_multi.msg import trackArray
from flir_adk_multi.msg import track1
from utils import CamObj
from utils import RadarObj
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError
import cv2
def main():
    rospy.init_node('jpda', anonymous=True)
    fusInst=jpda_class()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
class jpda_class:

    def __init__(self):
        self.TrackPub=rospy.Publisher("dataAssoc",trackArray, queue_size=100) 
        self.image_pub=rospy.Publisher("fusedImage",Image, queue_size=100) 
        self.YoloClassList=[0,1,2,3,5,7]
        self.GateThresh =2 # Scaling factor, threshold for gating
        self.bridge=CvBridge()
        # Initializing parameters:
        self.Q_rdr=np.array([[1,0,0,0],[0,1,0,0],[0,0,0.1,0],[0,0,0,0.1]])
        self.R_rdr=np.array([[1,0,0],[0,1,0],[0,0,0]])
        self.Vt=[]
        self.psi=[]
        self.psiD=[]# psiDot
        rospy.Subscriber('/cam_front/raw', Image, self.buildImage)
        rospy.Subscriber('/vel', Twist, self.Odom1) 
        rospy.Subscriber('/odom', Odometry, self.Odom2) 
        rospy.Subscriber('/imu', Imu, self.Odom3) 
        # rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.CamMsrmts)
        
        rospy.Subscriber('/radar_front', RadarObjects, self.RdrMsrmtsNuSc)
        
        # rospy.Subscriber('/radar_front', BoundingBoxes, self.RdrMsrmtsMKZ)
        #TODO: if-else switches for trackInitiator, trackDestructor, and Kalman functions for radar and camera
        
        # self.nearestNAlg()
        # self.JPDAalg()
    def buildImage(self,data):
        self.image=self.bridge.imgmsg_to_cv2(data, "bgr8")

    def Odom1(self,data):
        self.Vt =data.linear.x 
    def Odom2(self,data):
        self.psi=data.pose.pose.orientation.z
    def Odom3(self,data):
        self.psiD=data.angular_velocity.z


    def trackInitiator(self,SensorData):
        if isinstance(SensorData[0],CamObj): 
            pass
        elif isinstance(SensorData[0],RadarObj):
            if hasattr(self, 'CurrentTracks'):# Some (or Zer0) tracks already exists (i.e, not start of algorithm)
                # Move to current tracks based on NN-style gating
                toDel=[]

                for idx in range(len(self.InitiatedTracks.tracks)):
                    gateValX=[]
                    gateValY=[]
                    gateValRMS=[]
                    # Find all sensor objects within some gate
                    for jdx in range(len(SensorData)):
                        gateValX.append(np.abs(SensorData[jdx].pose.x-self.InitiatedTracks.tracks[idx].x))
                        gateValY.append(np.abs(SensorData[jdx].pose.y-self.InitiatedTracks.tracks[idx].y))
                        gateValRMS.append(np.sqrt(gateValX[jdx]**2+gateValY[jdx]**2))
                    if (np.min(gateValRMS)<=2.0): # @50Hz, 20m/s in X dir and 10m/s in Y-Direction as validation gate
                        #If gate is satisfied, move to CurrentTracks after initiating P
                        self.InitiatedTracks.tracks[idx].P=Mat_buildROS(np.array([[1,0,0,0],[0,1,0,0],[0,0,0.5,0],[0,0,0,2]]))
                        #(Large uncertainity given to Beta. Others conservatively picked based on Delphi ESR spec sheet)
                        self.InitiatedTracks.tracks[idx].Stat=1# Moving to CurrentTracks
                        x=self.InitiatedTracks.tracks[idx].x
                        y=self.InitiatedTracks.tracks[idx].y
                        Vc=self.InitiatedTracks.tracks[idx].Vc
                        Beta=self.InitiatedTracks.tracks[idx].B
                        psi=self.psi
                        psiD=self.psiD
                        Vt=self.Vt
                        posNorm=np.sqrt(x**2+y**2)
                        H31=(Vc*np.sin(psi-Beta)*y**2-x*y*(Vc*np.cos(psi-Beta)-Vt))/(posNorm**3)
                        H32=(-Vc*np.sin(psi-Beta)*x*y+x**2*(Vc*np.cos(psi-Beta)-Vt))/(posNorm**3)
                        H33=x*np.sin(psi-Beta)/posNorm+y*np.cos(psi-Beta)/posNorm
                        H34=(-x*Vc*np.cos(psi-Beta)+y*Vc*np.sin(psi-Beta))/posNorm
                        Hk=np.array([[1,0,0,0],[x/posNorm,y/posNorm,0,0],[H31,H32,H33,H34]])
                        self.InitiatedTracks.tracks[idx].H=Mat_buildROS(Hk)
                        self.CurrentTracks.tracks.append(self.InitiatedTracks.tracks[idx])
                        #Build Arrays for deletion:
                        toDel.append([idx])
                        #Also Delete the corresponding SensorData value:
                        np.delete(SensorData,np.argmin(gateValRMS))
                    else: # none of the SensorData is close to InitiatedTracks[idx], so delete it
                        toDel.append([idx])
                # Clean all InitiatedTracks with status 1
                np.delete(self.InitiatedTracks.tracks,toDel)
                # Then append remaining sensor Data for future initation
                for idx in range(len(SensorData)):
                    self.InitiatedTracks.tracks.append(track1())
                    self.InitiatedTracks.tracks[-1].Stat= -1 # InitiatedTrack
                    self.InitiatedTracks.tracks[-1].x=SensorData[idx].pose.x
                    self.InitiatedTracks.tracks[-1].y=SensorData[idx].pose.y
                    self.InitiatedTracks.tracks[-1].Vc=np.sqrt(SensorData[idx].vx_comp**2+SensorData[idx].vy_comp**2)
                    self.InitiatedTracks.tracks[-1].B=self.psi # TODO: Have better estimate for Beta during intialization

            else: # Start of algorithm, no tracks
                self.CurrentTracks=trackArray()
                print('Initiated CurrentTracks')
                self.InitiatedTracks=trackArray()
                for idx in range(len(SensorData)):
                    self.InitiatedTracks.tracks.append(track1())
                    self.InitiatedTracks.tracks[-1].Stat= -1 # InitiatedTrack
                    self.InitiatedTracks.tracks[-1].x=SensorData[idx].pose.x
                    self.InitiatedTracks.tracks[-1].y=SensorData[idx].pose.y
                    self.InitiatedTracks.tracks[-1].Vc=np.sqrt(SensorData[idx].vx_comp**2+SensorData[idx].vy_comp**2)
                    self.InitiatedTracks.tracks[-1].B=self.psi # TODO: Have better estimate for Beta during intialization
                

    def trackDestructor(self,SensorData):
        if isinstance(SensorData[0],CamObj): 
            pass
        elif isinstance(SensorData[0],RadarObj):
            toDel=[]
            for idx in range(len(self.CurrentTracks.tracks)):
                if self.CurrentTracks.tracks[idx].Stat>=14: # If no measurements associated for 5 steps
                    toDel.append([idx])
            np.delete(self.CurrentTracks.tracks,toDel)

    def trackMaintenance(self,SensorData):
        if isinstance(SensorData[0],CamObj): 
            pass
        elif isinstance(SensorData[0],RadarObj):
            SensorIndices=[]
            for idx in range(len(self.CurrentTracks.tracks)):
                SensorIndices.append(self.ValidationGate(SensorData,self.CurrentTracks.tracks[idx]))#Clean the incoming data
                # Above yields array of possible measurments (only indices) corresponding to a particular track
            self.CurrentTracks=self.KalmanEstimate(SensorData,SensorIndices) # Includes DataAssociation Calcs
            self.CurrentTracks=self.KalmanPropagate(SensorData)
            self.TrackPub.publish(self.CurrentTracks)
            rospy.loginfo('Current tracks published to topic /dataAssoc')
            

    def trackPlotter(self):
       
        n=len(self.CurrentTracks.tracks)
        RadarAnglesH=np.zeros((n,1))
        RadarAnglesV=np.zeros((n,1))
        # Camera Coordinates: X is horizontal, Y is vertical starting from left top corner
        CirClr=[]
        for idx in range(n):
            RadarAnglesH[idx]=-np.degrees(np.arctan(np.divide(self.CurrentTracks.tracks[idx].y,self.CurrentTracks.tracks[idx].x)))
            RadarAnglesV[idx]=np.abs(np.degrees(np.arctan(np.divide(1,self.CurrentTracks.tracks[idx].x)))) #will always be negative, so correct for it
            if self.CurrentTracks.tracks[idx].Stat==1: #Current Track
                CirClr.append(np.array([0,255,0]))
            elif self.CurrentTracks.tracks[idx].Stat<=0: # Candidate Tracks for initialization
                CirClr.append(np.array([255,0,0]))
            else: # Candidate for Destructor
                CirClr.append(np.array([0,0,255]))
        CameraX=RadarAnglesH*(self.image.shape[1]/70) + self.image.shape[1]/2 # Number of pixels per degree,adjusted for shifting origin from centerline to top left
        CameraY=RadarAnglesV*(self.image.shape[0]/39.375) +450 # Number of pixels per degree,adjusted for shifting origin from centerline to top left
        for idx in range(len(RadarAnglesH)):
            if (CameraX[idx]<=self.image.shape[1]):
                self.image=cv2.circle(self.image, (int(CameraX[idx]),int(CameraY[idx])), 3, CirClr[idx].tolist())
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.image, "bgr8"))
        rospy.loginfo('Image is being published')


    def trackManager(self,SensorData):
        self.trackInitiator(SensorData)
        self.trackMaintenance(SensorData)
        self.trackDestructor(SensorData)
        self.trackPlotter()
        

                
    def DataAssociation(self,SensorData,SensorIndices,Method):
        if Method=="Hungarian":
            pass
        elif Method=="JPDA":
            pass
        elif Method=="Greedy": # Simple method that just outputs the closest UNUSED measurement
            usedSensorIndices=[]
            Yk=[]
            for idx in range(len(self.CurrentTracks.tracks)):
                gateValX=[]
                gateValY=[]
                gateValRMS=[]
                if (len(SensorIndices[idx])==0):
                    Yk.append([])
                    continue
                else:
                    for jdx in range(len(SensorIndices[idx])):
                        # print(len(np.array([SensorIndices[jdx]]).flatten().astype(int))==0)
                        gateValX.append(np.abs(SensorData[np.array([SensorIndices[idx][jdx]]).flatten().astype(int)][0].pose.x-self.CurrentTracks.tracks[idx].x))
                        gateValY.append(np.abs(SensorData[np.array([SensorIndices[idx][jdx]]).flatten().astype(int)][0].pose.y-self.CurrentTracks.tracks[idx].y))
                        gateValRMS.append(np.sqrt(gateValX[jdx]**2+gateValY[jdx]**2))
                    sensIdx=int(np.argmin(gateValRMS))
                    while sensIdx in usedSensorIndices:
                        np.delete(gateValRMS,sensIdx)
                        sensIdx=int(np.argmin(gateValRMS))
                    usedSensorIndices.append(sensIdx)
                    Yk.append(SensorData[sensIdx])
        return Yk # An Array with same len as CurrentTracks.tracks[]

    def KalmanPropagate(self,SensorData):
        delT=(SensorData[0].header.stamp-self.CurrentTracks.header.stamp)/1e9 # from nano seconds
        if isinstance(SensorData[0],CamObj): 
            PropagatedTracks=self.CurrentTracks
            for idx in range(len(PropagatedTracks.tracks)):
                PropagatedTracks.tracks[idx].pose.x=PropagatedTracks.tracks[idx].twist.linear.x*delT
                PropagatedTracks.tracks[idx].pose.y=PropagatedTracks.tracks[idx].twist.linear.y*delT
        elif isinstance(SensorData[0],RadarObj):
            PropagatedTracks=self.CurrentTracks
            for idx in range(len(PropagatedTracks.tracks)):
                x=PropagatedTracks.tracks[idx].x
                y=PropagatedTracks.tracks[idx].y
                Vc=PropagatedTracks.tracks[idx].Vc
                Beta=PropagatedTracks.tracks[idx].B
                psi=self.psi
                psiD=self.psiD
                Vt=self.Vt
                F14=-delT*Vc*np.cos(psi-Beta)
                F24=delT*Vc*np.sin(psi-Beta)
                Fk=np.array([[1,delT*psiD,delT*np.sin(psi-Beta),F14],[-delT*psiD,1,delT*np.cos(psi-Beta),F24],[0,0,1,0],[0,0,0,1]])
                PropagatedTracks.tracks[idx].F=Mat_buildROS(Fk)
                PropagatedTracks.tracks[idx].P=Mat_buildROS(Fk*Mat_extractROS(PropagatedTracks.tracks[idx].P)*Fk.T+self.Q_rdr*(delT**2)/0.01)
                StateVec=np.array([PropagatedTracks.tracks[idx].x, PropagatedTracks.tracks[idx].y,PropagatedTracks.tracks[idx].Vc,PropagatedTracks.tracks[idx].B])
                A=np.array([[0,psiD,np.sin(psi-Beta),0],[-psiD,0,0,np.cos(psi-Beta)],[0,0,0,0],[0,0,0,0]])
                StateVec=StateVec+delT*(A*StateVec+np.array([[0],[Vt],[0],[0]]))
                PropagatedTracks.tracks[idx].x=StateVec[0]
                PropagatedTracks.tracks[idx].y=StateVec[1]
                PropagatedTracks.tracks[idx].Vc=StateVec[2]
                PropagatedTracks.tracks[idx].B=StateVec[3]
        return PropagatedTracks


    def KalmanEstimate(self,SensorData,SensorIndices):
        delT=(SensorData[0].header.stamp-self.CurrentTracks.header.stamp)/1e9 # from nano seconds
        if isinstance(SensorData[0],CamObj): 
            pass
        elif isinstance(SensorData[0],RadarObj): # Use EKF from Truck Platooning paper:
            EstimatedTracks=self.CurrentTracks
            Yk=self.DataAssociation(SensorData,SensorIndices,'Greedy') # Lists suitable measurements for each track
            for idx in range(len(EstimatedTracks.tracks)):
                if Yk[idx]==[]: # No suitable measurements found, move to potential destruct
                    if  EstimatedTracks.tracks[idx].Stat>=10:
                         EstimatedTracks.tracks[idx].Stat+=1
                    else:
                        EstimatedTracks.tracks[idx].Stat=10
                    continue
                else:
                    x=EstimatedTracks.tracks[idx].x
                    y=EstimatedTracks.tracks[idx].y
                    Vc=EstimatedTracks.tracks[idx].Vc
                    Beta=EstimatedTracks.tracks[idx].B
                    psi=self.psi
                    psiD=self.psiD
                    Vt=self.Vt
                    posNorm=np.sqrt(x**2+y**2)
                    H31=(Vc*np.sin(psi-Beta)*y**2-x*y*(Vc*np.cos(psi-Beta)-Vt))/(posNorm**3)
                    H32=(-Vc*np.sin(psi-Beta)*x*y+x**2*(Vc*np.cos(psi-Beta)-Vt))/(posNorm**3)
                    H33=x*np.sin(psi-Beta)/posNorm+y*np.cos(psi-Beta)/posNorm
                    H34=(-x*Vc*np.cos(psi-Beta)+y*Vc*np.sin(psi-Beta))/posNorm
                    Hk=np.array([[1,0,0,0],[x/posNorm,y/posNorm,0,0],[H31,H32,H33,H34]])
                    EstimatedTracks.tracks[idx].H=Mat_buildROS(Hk)
                    Pk=Mat_extractROS(EstimatedTracks.tracks[idx].P)
                    K=Pk*Hk.T*np.linalg.inv(Hk*P*Hk.T+self.R_rdr)
                    EstimatedTracks.tracks[idx].K=Mat_buildROS(K)
                    StateVec=np.array([EstimatedTracks.tracks[idx].x, EstimatedTracks.tracks[idx].y,EstimatedTracks.tracks[idx].Vc,EstimatedTracks.tracks[idx].B])
                    StateVec=StateVec+K*(Yk[idx]-Hk*StateVec)
                    EstimatedTracks.tracks[idx].x=StateVec[0]
                    EstimatedTracks.tracks[idx].y=StateVec[1]
                    EstimatedTracks.tracks[idx].Vc=StateVec[2]
                    EstimatedTracks.tracks[idx].B=StateVec[3]
                    Pk=(np.eye(4)-K*Hk)*Pk
                    EstimatedTracks.tracks[idx].P=Mat_buildROS(Pk)
        return EstimatedTracks



    def ValidationGate(self,SensorData,track):
        SensorIdxOut=[]
        if isinstance(SensorData[0],CamObj): #
            pass
        if isinstance(SensorData[0],RadarObj):
            StateVec=np.array([track.x, track.y,track.Vc,track.B])
            Hk=Mat_extractROS(track.H)
            y_est=np.dot(Hk.reshape(3,4),StateVec.reshape(4,1))
            Pk=Mat_extractROS(track.P)
            SkInv=np.linalg.inv((Hk.dot(Pk)).dot(Hk.T)+self.R_rdr)
            # TODO: Edit for Delphi ESR
            for jdx in range(len(SensorData)):
                Vc=np.sqrt((self.Vt+SensorData[jdx].vx)**2+SensorData[jdx].vy**2)
                Beta=  SensorData[jdx].vy_comp/SensorData[jdx].vx_comp# This will be Vx/Vy for delphi esr
                rho=np.sqrt(SensorData[jdx].pose.x**2+SensorData[jdx].pose.y**2)
                rhoDot=(SensorData[jdx].pose.x*np.sin(self.psi-Beta)*Vc+SensorData[jdx].pose.y*np.cos(self.psi-Beta)*Vc)/rho
                y=np.array([SensorData[jdx].pose.x,rho,rhoDot])
                Temp=((y.reshape(3,1)-y_est).T.dot(SkInv)).dot(y.reshape(3,1)-y_est)
                if (Temp[0]<=self.GateThresh**2):
                    SensorIdxOut.append([jdx])
        return SensorIdxOut

    def CamMsrmts(self,data):
        self.CamReadings=[]
        pass
        # for idx in range(len(data.bounding_boxes)):
        #     #print(data.bounding_boxes[2].id)
        #     if data.bounding_boxes[idx].id in self.YoloClassList:
        #         self.CamReadings.append(CamObj())
        #         self.CamReadings[-1].stamp=data.header.stamp
        #         self.CamReadings[-1].PxWidth=(data.bounding_boxes[idx].xmax-data.bounding_boxes[idx].xmin)
        #         self.CamReadings[-1].PxHeight=(data.bounding_boxes[idx].ymax-data.bounding_boxes[idx].ymin)
        #         self.CamReadings[-1].id=data.bounding_boxes[idx].id
        #self.trackManager(self.CamReadings)
        # for elem in self.TempReadings:
        #     if elem:
        #         self.CamReadings.append(elem)
        #print(len(self.CamReadings))
    #TODO: create new RdrMsrmtsMKZ to work with MKZ.
    #def RdrMsrmtsNuSc(self,data): 

    def RdrMsrmtsNuSc(self,data): 
        #Build SensorData
        self.RdrReadings=[]
        
        for idx in range(len(data.objects)):
            self.RdrReadings.append(RadarObj())
            self.RdrReadings[-1].pose=data.objects[idx].pose
            self.RdrReadings[-1].vx=data.objects[idx].vx
            self.RdrReadings[-1].vy=data.objects[idx].vy
            self.RdrReadings[-1].vx_comp=data.objects[idx].vx_comp
            self.RdrReadings[-1].vy_comp=data.objects[idx].vy_comp
            self.RdrReadings[-1].header=data.header
            
        #print(self.RdrReadings[3].stamp)
        self.RdrReadings=np.asarray(self.RdrReadings)
        self.trackManager(self.RdrReadings)
        


if __name__=='__main__':

	main()
    

            
        
