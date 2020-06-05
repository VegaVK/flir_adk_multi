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
from utils import Mat_buildROS
from utils import Mat_extractROS
from flir_adk_multi.msg import trackArray
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
def main():
    rospy.init_node('jpda', anonymous=True)
    fusInst=jpda_class()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
class jpda_class:

    def __init__(self):
        self.TrackPub=rospy.Publisher("JPDA",trackArray, queue_size=100) 
        self.YoloClassList=[0,1,2,3,5,7]
        self.GateThresh =2 # Scaling factor, threshold for gating
        # Initializing parameters:
        self.Q_rdr=np.array([[1,0,0,0],[0,1,0,0],[0,0,0.1,0],[0,0,0,0.1]])
        self.R_rdr=np.array([[1,0,0],[0,1,0],[0,0,0]])
        
        rospy.Subscriber('/odom', BoundingBoxes, self.Odom) # Placeholder
        rospy.Subscriber('darknet_ros/bounding_boxes', BoundingBoxes, self.CamMsrmts)
        rospy.Subscriber('/radar_front', RadarObjects, self.RdrMsrmtsNuSc)
        # rospy.Subscriber('/radar_front', BoundingBoxes, self.RdrMsrmtsMKZ)
        #TODO: if-else switches for trackInitiator, trackDestructor, and Kalman functions for radar and camera
        
        # self.nearestNAlg()
        # self.JPDAalg()

    def Odom(self,data):
        pass # Needs psi, psiDot and Vt

    def trackInitiator(self,SensorData):
        if self.CurrentTracks in locals():
            #Check new measurements and initiate candiates
        else:
            self.CurrentTracks=[]

    def trackDestructor(self,SensorData):
        pass


    def trackManager(self,SensorData):
        self.trackInitiator(SensorData)
        self.NewValidMsmts=self.ValidationGate(data1,self.CurrentTracks)
        self.trackDestructor(SensorData)
        



                
    def DataAssociation(self,CurrentTracks,SensorData,Method):
        if Method=="NN":
            pass
        elif Method=="JPDA":
            pass

        return Yk # An Array with same len as CurrentTracks.tracks[]

    def KalmanPropagate(self,CurrentTracks,SensorData):
        delT=(SensorData[0].header.stamp-CurrentTracks.header.stamp)/1e9 # from nano seconds
        if isinstance(SensorData[0],CamObj): 
            PropagatedTracks=CurrentTracks
            for idx in len(PredictedTracks.tracks):
                PropagatedTracks.tracks[idx].pose.x=PredictedTracks.tracks[idx].twist.linear.x*delT
                PropagatedTracks.tracks[idx].pose.y=PredictedTracks.tracks[idx].twist.linear.y*delT
            return PropagatedTracks

        elif isinstance(SensorData[0],RadarObj):
            PropagatedTracks=CurrentTracks
            for idx in len(PropagatedTracks.tracks):
                x=PropagatedTracks.tracks[idx].x
                y=PropagatedTracks.tracks[idx].y
                Vc=PropagatedTracks.tracks[idx].Vc
                Beta=PropagatedTracks.tracks[idx].B
                psi=self.psi
                psiD=self.psiDot
                Vt=self.Vt
                F14=-delT*Vc*np.cos(psi-Beta)
                F24=delT*Vc*np.sin(psi-Beta)
                Fk=np.array([[1,delT*psiD,delT*np.sin(psi-Beta),F14],[-delT*psiD,1,delT*np.cos(psi-Beta),F24],[0,0,1,0],[0,0,0,1]])
                PropagatedTracks.tracks[idx].F=Mat_buildROS(Fk)
                PropagatedTracks.tracks[idx].P=Mat_buildROS(Fk*Mat_extractROS(PropagatedTracks.tracks[idx].P)*Fk.T+self.Q_rdr*(delT^2)/0.01)


            


    def KalmanEstimate(self,CurrentTracks,SensorData):
        delT=(SensorData[0].header.stamp-CurrentTracks.header.stamp)/1e9 # from nano seconds
        if isinstance(SensorData[0],CamObj): 
            pass
        
        elif isinstance(SensorData[0],RadarObj): # Use EKF from Truck Platooning paper:
            EstimatedTracks=CurrentTracks
            Yk=self.DataAssociation(SensorData)
            for idx in len(EstimatedTracks.tracks):
                x=EstimatedTracks.tracks[idx].x
                y=EstimatedTracks.tracks[idx].y
                Vc=EstimatedTracks.tracks[idx].Vc
                Beta=EstimatedTracks.tracks[idx].B
                psi=self.psi
                psiD=self.psiDot
                Vt=self.Vt
                posNorm=np.sqrt(x^2+y^2)
                H31=(Vc*np.sin(psi-Beta)*y^2-x*y*(Vc*np.cos(psi-Beta)-Vt))/(posNorm^3)
                H32=(-Vc*np.sin(psi-Beta)*x*y+x^2*(Vc*np.cos(psi-Beta)-Vt))/(posNorm^3)
                H33=x*np.sin(psi-Beta)/posNorm+y*np.cos(psi-Beta)/posNorm
                H34=(-x*Vc*np.cos(psi-Beta)+y*Vc*np.sin(psi-Beta))/posNorm
                Hk=np.array([[1,0,0,0],[x/posNorm,y/posNorm,0,0],[H31,H32,H33,H34]])
                EstimatedTracks.tracks[idx].H=Mat_buildROS(Hk)
                Pk=Mat_extractROS(EstimatedTracks.tracks[idx].P)
                K=Pk*Hk.T*np.inv(Hk*P*Hk.T+self.R_rdr)
                EstimatedTracks.tracks[idx].K=Mat_buildROS(K)
                StateVec=np.array([EstimatedTracks.tracks[idx].x, EstimatedTracks.tracks[idx].y,EstimatedTracks.tracks[idx].Vc,EstimatedTracks.tracks[idx].B])
                StateVec=StateVec+K*(Yk[idx]-Hk*StateVec)
                EstimatedTracks.tracks[idx].x=StateVec[0]
                EstimatedTracks.tracks[idx].y=StateVec[1]
                EstimatedTracks.tracks[idx].Vc=StateVec[2]
                EstimatedTracks.tracks[idx].B=StateVec[3]
                Pk=(np.eye(4)-K*Hk)*Pk
                EstimatedTracks.tracks[idx].P=Mat_buildROS(Pk)



    def ValidationGate(self,SensorData,CurrentTracks):
        if isinstance(SensorData[0],CamObj): #
            PredictedTracks=self.KalmanPredictor(CurrentTracks,SensorData)
            ValidatedMsmts=np.empty(len(PredictedTracks))
            for idx in range(len(PredictedTracks)):
                for jdx in range(len(SensorData))
                    if (PredictedTracks-SensorData[jdx]).T*np.inv(S)*(PredictedTracks-SensorData[jdx])<=self.GateThresh^2:
                        ValidatedMsmts[idx].append(SensorData[jdx])
            return ValidatedMsmts
    
        
                
            


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
        self.trackManager(self.CamReadings)
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
            self.RdrReadings[-1].vx=data.objects[idx].vx_comp
            self.RdrReadings[-1].vy=data.objects[idx].vy_comp
            self.RdrReadings[-1].stamp=data.header.stamp
        #print(self.RdrReadings[3].stamp)
        self.trackManager(self.RdrReadings)
        


if __name__=='__main__':

	main()
    

            
        
