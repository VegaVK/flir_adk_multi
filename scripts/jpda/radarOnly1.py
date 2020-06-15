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
from flir_adk_multi.msg import trackArray
from flir_adk_multi.msg import track1
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
    inputArg=sys.argv[1]
    # print(inputArg)
    fusInst=jpda_class(inputArg)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
class jpda_class():

    def __init__(self,inputArg):
        self.TrackPub=rospy.Publisher("dataAssoc",trackArray) 
        self.image_pub=rospy.Publisher("fusedImage",Image, queue_size=1000) 
        self.YoloClassList=[0,1,2,3,5,7]
        self.GateThresh =2# Scaling factor, threshold for gating
        self.trackInitThresh=1 # For track initiation
        self.bridge=CvBridge()
        self.image=[] # Initialized
        # Initializing parameters:
        self.Q_rdr=np.array([[1,0,0,0],[0,1,0,0],[0,0,0.1,0],[0,0,0,0.1]])
        self.R_rdr=np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.Vt=[]
        self.velY=[]
        self.velX=[]
        self.psi=[]
        self.psiD=[]# psiDot
        self.CamXOffset=2.36#=93 inches, measured b/w cam and Rdr, in x direction
        self.CamZoffset=1 # Roughly 40 inches
        if inputArg=="NuSc":
            rospy.Subscriber('/cam_front/raw', Image, self.buildImage)
            rospy.Subscriber('/vel', Twist, self.Odom1NuSc) 
            rospy.Subscriber('/odom', Odometry, self.Odom2NuSc) 
            rospy.Subscriber('/imu', Imu, self.Odom3NuSc)
            rospy.Subscriber('/radar_front', BoundingBoxes, self.RdrMsrmtsNuSc)
        elif inputArg=="MKZ":
            rospy.Subscriber('/Thermal_Panorama', Image, self.buildImage)
            rospy.Subscriber('/vehicle/steering_report', SteeringReport, self.Odom1MKZ) 
            rospy.Subscriber('/vehicle/gps/vel', TwistStamped, self.Odom2MKZ) # TODO: fix after IMU is available
            rospy.Subscriber('/vehicle/twist', TwistStamped,self.Odom3MKZ)
            rospy.Subscriber('/as_tx/objects', ObjectWithCovarianceArray, self.RdrMsrmtsMKZ)
        # rospy.Subscriber('/as_tx/objects')
        # rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.CamMsrmts)
        # rospy.Subscriber('/radar_front', BoundingBoxes, self.RdrMsrmtsMKZ)
        #TODO: if-else switches for trackInitiator, trackDestructor, and Kalman functions for radar and camera
        
        # self.nearestNAlg()
        # self.JPDAalg()
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
        if isinstance(SensorData[0],CamObj): 
            pass
        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
            
            if hasattr(self, 'CurrentTracks'):# Some (or Zer0) tracks already exists (i.e, not start of algorithm)
                # Move to current tracks based on NN-style gating
                toDel=[]

                for idx in range(len(self.InitiatedTracks.tracks)):
                    gateValX=[]
                    gateValY=[]
                    gateValRMS=[]
                    # Find all sensor objects within some gate
                    for jdx in range(len(SensorData)):
                        gateValX.append(np.abs(SensorData[jdx].pose.position.x-self.InitiatedTracks.tracks[idx].x.data))
                        gateValY.append(np.abs(SensorData[jdx].pose.position.y-self.InitiatedTracks.tracks[idx].y.data))
                        gateValRMS.append(np.sqrt((gateValX[jdx])**2+(gateValY[jdx])**2))
                    if (np.min(np.array(gateValRMS))<=self.trackInitThresh): # @50Hz, 20m/s in X dir and 10m/s in Y-Direction as validation gate
                        #If gate is satisfied, move to CurrentTracks after initiating P
                        self.InitiatedTracks.tracks[idx].P=Mat_buildROS(np.array([[1,0,0,0],[0,1,0,0],[0,0,0.5,0],[0,0,0,2]]))
                        #(Large uncertainity given to Beta. Others conservatively picked based on Delphi ESR spec sheet)
                        self.InitiatedTracks.tracks[idx].Stat.data=1# Moving to CurrentTracks
                        x=self.InitiatedTracks.tracks[idx].x.data
                        y=self.InitiatedTracks.tracks[idx].y.data
                        Vc=self.InitiatedTracks.tracks[idx].Vc.data
                        Beta=self.InitiatedTracks.tracks[idx].B.data
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
                        self.CurrentTracks.header=self.InitiatedTracks.header
                        self.CurrentTracks.tracks=np.append(self.CurrentTracks.tracks,self.InitiatedTracks.tracks[idx])
                        #Build Arrays for deletion:
                        toDel.append(idx)
                        #Also Delete the corresponding SensorData value:
                        SensorData=np.delete(SensorData,np.argmin(gateValRMS))
                    else: # none of the SensorData is close to InitiatedTracks[idx], so delete it
                        toDel.append(idx)
                # Clean all InitiatedTracks with status 1
                self.InitiatedTracks.tracks=np.delete(self.InitiatedTracks.tracks,toDel)
                # Then concatenate remaining sensor Data for future initation
                self.InitiatedTracks.header=SensorData[0].header
                for idx in range(len(SensorData)):
                    self.InitiatedTracks.tracks=np.append(self.InitiatedTracks.tracks,track1())
                    self.InitiatedTracks.tracks[-1].Stat.data= -1 # InitiatedTrack
                    self.InitiatedTracks.tracks[-1].x.data=SensorData[idx].pose.position.x
                    self.InitiatedTracks.tracks[-1].y.data=SensorData[idx].pose.position.y
                    self.InitiatedTracks.tracks[-1].Vc.data=np.sqrt(SensorData[idx].vx_comp**2+SensorData[idx].vy_comp**2)
                    self.InitiatedTracks.tracks[-1].B.data=self.psi # TODO: Have better estimate for Beta during intialization

            else: # Start of algorithm, no tracks
                self.CurrentTracks=trackArray()
                # print('createdCurrentTracks')
                self.InitiatedTracks=trackArray()
                self.InitiatedTracks.header=SensorData[0].header
                for idx in range(len(SensorData)):
                    self.InitiatedTracks.tracks=np.append(self.InitiatedTracks.tracks,track1())
                    self.InitiatedTracks.tracks[-1].Stat.data= -1 # InitiatedTrack
                    self.InitiatedTracks.tracks[-1].x.data=SensorData[idx].pose.position.x
                    self.InitiatedTracks.tracks[-1].y.data=SensorData[idx].pose.position.y
                    self.InitiatedTracks.tracks[-1].Vc.data=np.sqrt(SensorData[idx].vx_comp**2+SensorData[idx].vy_comp**2)
                    self.InitiatedTracks.tracks[-1].B.data=self.psi # TODO: Have better estimate for Beta during intialization
                

    def trackDestructor(self,SensorData):
        if isinstance(SensorData[0],CamObj): 
            pass
        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
            toDel=[]
            for idx in range(len(self.CurrentTracks.tracks)):
                if self.CurrentTracks.tracks[idx].Stat.data>=12: # If no measurements associated for 5 steps
                    toDel.append(idx)
            self.CurrentTracks.tracks=np.delete(self.CurrentTracks.tracks,toDel)

    def trackMaintenance(self,SensorData):
        if isinstance(SensorData[0],CamObj): 
            pass
        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
            SensorIndices=[]
            for idx in range(len(self.CurrentTracks.tracks)):
                SensorIndices.append(self.ValidationGate(SensorData,self.CurrentTracks.tracks[idx]))#Clean the incoming data - outputs 2D python array
                # Above yields array of possible measurments (only indices) corresponding to a particular track
            self.KalmanEstimate(SensorData,SensorIndices) # Includes DataAssociation Calcs
            # print('done KEST')
            self.KalmanPropagate(SensorData)
            # print(len(self.CurrentTracks.tracks))
            self.TrackPub.publish(tracks =self.CurrentTracks.tracks)
            rospy.loginfo('Current tracks published to topic /dataAssoc')
            

    def trackPlotter(self):
        if len(self.image)==0:
           return # Skip function call if image is not available
        n=len(self.CurrentTracks.tracks)
        RadarAnglesH=np.zeros((n,1))
        RadarAnglesV=np.zeros((n,1))
        # Camera Coordinates: X is horizontal, Y is vertical starting from left top corner
        CirClr=[]
        for idx in range(n):
            RadarAnglesH[idx]=-np.degrees(np.arctan(np.divide(self.CurrentTracks.tracks[idx].y.data,self.CurrentTracks.tracks[idx].x.data)))
            RadarAnglesV[idx]=np.abs(np.degrees(np.arctan(np.divide(self.CamZoffset,self.CurrentTracks.tracks[idx].x.data+self.CamXOffset)))) #will always be negative, so correct for it
            if self.CurrentTracks.tracks[idx].Stat.data==1: #Current Track- green
                print('CURRENT TRACK!!!')
                CirClr.append(np.array([0,255,0]))
                print(CirClr[idx])
            elif self.CurrentTracks.tracks[idx].Stat.data<=0: # Candidate Tracks for initialization - blue
                CirClr.append(np.array([255,0,0]))
            else: # Candidate for Destructor-orange
                CirClr.append(np.array([0,165,255])) 
        CameraX=RadarAnglesH*(self.image.shape[1]/190) + self.image.shape[1]/2 # Number of pixels per degree,adjusted for shifting origin from centerline to top left
        CameraY=RadarAnglesV*(self.image.shape[0]/39.375) +512/2 # Number of pixels per degree,adjusted for shifting origin from centerline to top left
        CirClr=np.array(CirClr)
        for idx in range(len(RadarAnglesH)):
            if (CameraX[idx]<=self.image.shape[1]):
                self.image=cv2.circle(self.image, (int(CameraX[idx]),int(CameraY[idx])), 12, CirClr[idx].tolist(),3)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.image, "bgr8"))
        rospy.loginfo('Image is being published')


    def trackManager(self,SensorData):
        self.trackInitiator(SensorData)
        # print('done TrackInit')
        self.trackMaintenance(SensorData)
        # print('doneTrackMaint')
        self.trackDestructor(SensorData)
        # print('done TrackDestr')
        self.trackPlotter()
        print('len of CurrentTracks.tracks: ')
        print(len(self.CurrentTracks.tracks))
        

                
    def DataAssociation(self,SensorData,SensorIndices,Method):
        if Method=="Hungarian":
            pass
        elif Method=="JPDA":
            #Build A Validation Matrix if there are sufficient sensor data and tracks
            if (len(SensorData)<1) or (len(self.CurrentTracks.tracks)<1):
                Yk=[]
            else:
                ValidationMat=np.zeros((len(SensorData),len(self.CurrentTracks.tracks)+1),dtype=float)
                ValidationMat[:,0]=np.ones((len(SensorData),1))[:,0]
                # for idx in range(len(SensorData)): #Rows
                #     for jdx in range(len(SensorIndices[idx])):
                #         ValidationMat[idx][jdx+1]=1
                # Now select different permutations:
                seedList=np.append(np.zeros((1,len(SensorData)-1)),1)
                l=list(permutations(seedList.tolist())) ## TODO: PREVENT THIS FROM CRASHING THE CODE!!
                L=np.array(l).T
                
        elif Method=="Greedy": # Simple method that just outputs the closest UNUSED measurement
            # Sensor indices is a 2D python list, not numpy array
            usedSensorIndices=[]
            Yk=[] # A python list of sensor measurements corresponding to each CurrentTrack
            # print('len of CurrentTracks.tracks: ')
            # print(len(self.CurrentTracks.tracks))
            for idx in range(len(self.CurrentTracks.tracks)):
                # print('dataAssoc Yk idx:')
                # print(idx)
                gateValX=[]
                gateValY=[]
                gateValRMS=[]
                # print(not(SensorIndices[idx]))
                if len(SensorIndices[idx])==0:
                    Yk.append([])
                    continue
                else:
                    for jdx in range(len(SensorIndices[idx])):
                        gateValX.append(np.abs(SensorData[SensorIndices[idx][jdx]].pose.position.x-self.CurrentTracks.tracks[idx].x.data))
                        gateValY.append(np.abs(SensorData[SensorIndices[idx][jdx]].pose.position.y-self.CurrentTracks.tracks[idx].y.data))
                        gateValRMS.append(np.sqrt((gateValX[jdx])**2+(gateValY[jdx])**2))
                    if np.min(gateValRMS)<=self.GateThresh:
                        sensIdx=int(np.argmin(np.array(gateValRMS)))
                        gateValRMS=np.array(gateValRMS)
                        temp=SensorData[sensIdx]
                        while sensIdx in usedSensorIndices:
                            # print('while sensIdx and usedSensorIndices and gateValRSM:')
                            # print(sensIdx)
                            # print(usedSensorIndices)
                            # print(gateValRMS)
                            gateValRMS=np.delete(gateValRMS,sensIdx)
                            # print(gateValRMS)
                            if len(gateValRMS)==0:
                                temp=[]
                                # print('going to break')
                                break
                            sensIdx=int(np.argmin(np.array(gateValRMS)))
                            temp=SensorData[sensIdx]
                            # print('sensIdx:')
                            # print(sensIdx)
                        usedSensorIndices.append(sensIdx)
                        Yk.append(temp)
                    else:
                        Yk.append([])
        return Yk # An Array with same len as CurrentTracks.tracks[]
        # done('dataAssoc')

    def KalmanPropagate(self,SensorData):
        delT=(SensorData[0].header.stamp-self.CurrentTracks.header.stamp) 
        delT=delT.to_sec()
        if isinstance(SensorData[0],CamObj):
            pass 
            # for idx in range(len(self.CurrentTracks.tracks)):
                # self.CurrentTracks.tracks[idx].pose.x=self.CurrentTracks.tracks[idx].twist.linear.x*delT
                # self.CurrentTracks.tracks[idx].pose.y=self.CurrentTracks.tracks[idx].twist.linear.y*delT
        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
            for idx in range(len(self.CurrentTracks.tracks)):
                x=self.CurrentTracks.tracks[idx].x.data
                y=self.CurrentTracks.tracks[idx].y.data
                Vc=self.CurrentTracks.tracks[idx].Vc.data
                Beta=self.CurrentTracks.tracks[idx].B.data
                psi=self.psi
                psiD=self.psiD
                Vt=self.Vt
                F14=-delT*Vc*np.cos(psi-Beta)
                F24=delT*Vc*np.sin(psi-Beta)
                Fk=np.array([[1,delT*psiD,delT*np.sin(psi-Beta),F14],[-delT*psiD,1,delT*np.cos(psi-Beta),F24],[0,0,1,0],[0,0,0,1]])
                self.CurrentTracks.tracks[idx].F=Mat_buildROS(Fk)
                self.CurrentTracks.tracks[idx].P=Mat_buildROS(Fk*Mat_extractROS(self.CurrentTracks.tracks[idx].P)*Fk.T+self.Q_rdr*(delT**2)/0.01)
                StateVec=np.array([x, y,Vc,Beta])
                A=np.array([[0,psiD,np.sin(psi-Beta),0],[-psiD,0,0,np.cos(psi-Beta)],[0,0,0,0],[0,0,0,0]])
                StateVec=StateVec.reshape(4,1)+delT*(A.dot(StateVec.reshape(4,1))+np.array([[0],[Vt],[0],[0]]))
                self.CurrentTracks.tracks[idx].x.data=StateVec[0]
                self.CurrentTracks.tracks[idx].y.data=StateVec[1]
                self.CurrentTracks.tracks[idx].Vc.data=StateVec[2]
                self.CurrentTracks.tracks[idx].B.data=StateVec[3]


    def KalmanEstimate(self,SensorData,SensorIndices):
        # print('reached KEST')
        if isinstance(SensorData[0],CamObj): 
            pass
        elif isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ): # Use EKF from Truck Platooning paper:
            # print('reached KEST2')
            Yk=self.DataAssociation(SensorData,SensorIndices,'Greedy') # Lists suitable measurements for each track
            # print('done dataAssoc KEST')
            for idx in range(len(Yk)):
                # print('Yk Idx:')
                # print(idx)
                if not Yk[idx]: # No suitable measurements found, move to potential destruct
                    if  self.CurrentTracks.tracks[idx].Stat.data>=10:
                         self.CurrentTracks.tracks[idx].Stat.data+=1
                    else:
                        self.CurrentTracks.tracks[idx].Stat.data=10
                    continue
                else:
                    x=self.CurrentTracks.tracks[idx].x.data
                    y=self.CurrentTracks.tracks[idx].y.data
                    Vc=self.CurrentTracks.tracks[idx].Vc.data
                    Beta=self.CurrentTracks.tracks[idx].B.data
                    psi=self.psi
                    psiD=self.psiD
                    Vt=self.Vt
                    posNorm=np.sqrt(x**2+y**2)
                    H31=(Vc*np.sin(psi-Beta)*y**2-x*y*(Vc*np.cos(psi-Beta)-Vt))/(posNorm**3)
                    H32=(-Vc*np.sin(psi-Beta)*x*y+x**2*(Vc*np.cos(psi-Beta)-Vt))/(posNorm**3)
                    H33=x*np.sin(psi-Beta)/posNorm+y*np.cos(psi-Beta)/posNorm
                    H34=(-x*Vc*np.cos(psi-Beta)+y*Vc*np.sin(psi-Beta))/posNorm
                    Hk=np.array([[1,0,0,0],[x/posNorm,y/posNorm,0,0],[H31,H32,H33,H34]])
                    self.CurrentTracks.tracks[idx].H=Mat_buildROS(Hk)
                    Pk=Mat_extractROS(self.CurrentTracks.tracks[idx].P)
                    K=np.dot(np.dot(Pk,Hk.T),np.linalg.inv(np.dot(np.dot(Hk,Pk),Hk.T)+self.R_rdr))
                    self.CurrentTracks.tracks[idx].K=Mat_buildROS(K)
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
                    self.CurrentTracks.tracks[idx].x.data=StateVec[0]
                    self.CurrentTracks.tracks[idx].y.data=StateVec[1]
                    self.CurrentTracks.tracks[idx].Vc.data=StateVec[2]
                    self.CurrentTracks.tracks[idx].B.data=StateVec[3]
                    Pk=np.dot((np.eye(4)-np.dot(K,Hk)),Pk)
                    self.CurrentTracks.tracks[idx].P=Mat_buildROS(Pk)



    def ValidationGate(self,SensorData,track):
        SensorIdxOut=[]
        if isinstance(SensorData[0],CamObj): #
            pass
        if isinstance(SensorData[0],RadarObj) or isinstance(SensorData[0],RadarObjMKZ):
            StateVec=np.array([track.x.data, track.y.data,track.Vc.data,track.B.data])
            Hk=Mat_extractROS(track.H)
            # print(StateVec.reshape(4,1)[1])
            y_est=np.dot(Hk.reshape(3,4),StateVec.reshape(4,1))
            Pk=Mat_extractROS(track.P)
            # print('SkInv:')
            # print(np.dot(np.dot(Hk,Pk),Hk.T)+self.R_rdr)
            SkInv=np.linalg.inv(np.dot(np.dot(Hk,Pk),Hk.T)+self.R_rdr)

            # TODO: Edit for Delphi ESR
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
                if (Temp[0]<=self.GateThresh**2):
                    SensorIdxOut.append(jdx)
        return SensorIdxOut # returns a python list, not numpy array

    def CamMsrmts(self,data):
        self.CamReadings=[]
        pass
        # for idx in range(len(data.bounding_boxes)):
        #     #print(data.bounding_boxes[2].id)
        #     if data.bounding_boxes[idx].id in self.YoloClassList:
        #         self.CamReadings.concatenate(CamObj())
        #         self.CamReadings[-1].stamp=data.header.stamp
        #         self.CamReadings[-1].PxWidth=(data.bounding_boxes[idx].xmax-data.bounding_boxes[idx].xmin)
        #         self.CamReadings[-1].PxHeight=(data.bounding_boxes[idx].ymax-data.bounding_boxes[idx].ymin)
        #         self.CamReadings[-1].id=data.bounding_boxes[idx].id
        #self.trackManager(self.CamReadings)
        # for elem in self.TempReadings:
        #     if elem:
        #         self.CamReadings.concatenate(elem)
        #print(len(self.CamReadings))
    #TODO: create new RdrMsrmtsMKZ to work with MKZ.

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
            
        #print(self.RdrReadings[3].stamp)
        self.RdrReadings=np.asarray(self.RdrReadings)
        self.trackManager(self.RdrReadings)


if __name__=='__main__':

	main()
    

        