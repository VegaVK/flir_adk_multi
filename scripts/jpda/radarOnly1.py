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
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
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
        self.Vt=[]
        self.psi=[]
        self.psiD=[]# psiDot
        
        rospy.Subscriber('/Vel', Twist, self.Odom1) 
        rospy.Subscriber('/odom', Odometry, self.Odom2) 
        rospy.Subscriber('/imu', Imu, self.Odom3) 
        rospy.Subscriber('darknet_ros/bounding_boxes', BoundingBoxes, self.CamMsrmts)
        rospy.Subscriber('/radar_front', RadarObjects, self.RdrMsrmtsNuSc)
        # rospy.Subscriber('/radar_front', BoundingBoxes, self.RdrMsrmtsMKZ)
        #TODO: if-else switches for trackInitiator, trackDestructor, and Kalman functions for radar and camera
        
        # self.nearestNAlg()
        # self.JPDAalg()

    def Odom1(self,data):
        self.Vt =data.linear.x 
    def Odom2(self,data):
        self.psi=data.pose.pose.orientation.z
    def Odom3(self,data):
        self.psiD=data.angular_velocity.z


    def trackInitiator(self,SensorData):
        if self.CurrentTracks in locals():
            #Check new measurements and initiate candiates
        else:
            pass

    def trackDestructor(self,SensorData):
        pass

    def trackMaintenance(self,SensorData):
        pass

    def trackManager(self,SensorData):
        SensorData=self.ValidationGate(SensorData) #Clean the incoming data
        self.trackInitiator(SensorData)
        self.trackMaintenance(SensorData)
        self.trackDestructor(SensorData)

                
    def DataAssociation(self,SensorData,Method):
        if Method=="NN":
            pass
        elif Method=="JPDA":
            pass

        return Yk # An Array with same len as CurrentTracks.tracks[]

    def KalmanPropagate(self,SensorData):
        delT=(SensorData[0].header.stamp-self.CurrentTracks.header.stamp)/1e9 # from nano seconds
        if isinstance(SensorData[0],CamObj): 
            PropagatedTracks=self.CurrentTracks
            for idx in len(PredictedTracks.tracks):
                PropagatedTracks.tracks[idx].pose.x=PredictedTracks.tracks[idx].twist.linear.x*delT
                PropagatedTracks.tracks[idx].pose.y=PredictedTracks.tracks[idx].twist.linear.y*delT
            return PropagatedTracks
        elif isinstance(SensorData[0],RadarObj):
            PropagatedTracks=self.CurrentTracks
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
                StateVec=np.array([PropagatedTracks.tracks[idx].x, PropagatedTracks.tracks[idx].y,PropagatedTracks.tracks[idx].Vc,PropagatedTracks.tracks[idx].B])
                A=np.array([[0,psiD,np.sin(psi-Beta),0],[-psiD,0,0,np.cos(psi-Beta)],[0,0,0,0],[0,0,0,0]])
                StateVec=StateVec+delT*(A*StateVec+np.array([0,Vt,0,0]).T)


    def KalmanEstimate(self,SensorData):
        delT=(SensorData[0].header.stamp-self.CurrentTracks.header.stamp)/1e9 # from nano seconds
        if isinstance(SensorData[0],CamObj): 
            pass
        elif isinstance(SensorData[0],RadarObj): # Use EKF from Truck Platooning paper:
            EstimatedTracks=self.CurrentTracks
            Yk=self.DataAssociation(SensorData,'NN')#Set of measurements associated with self.CurrentTracks; array of same size
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



    def ValidationGate(self,SensorData):
        SensorOutData=[]
        if isinstance(SensorData[0],CamObj): #
            pass
        if isinstance(SensorData[0],RadarObj):
            for idx in range(len(self.CurrentTracks.tracks)):
                StateVec=np.array([self.CurrentTracks.tracks[idx].x, self.CurrentTracks.tracks[idx].y,self.CurrentTracks.tracks[idx].Vc,self.CurrentTracks.tracks[idx].B])
                y_est[idx]=Mat_extractROS(self.CurrentTracks.tracks[idx].H)*StateVec
                Hk=Mat_extractROS(self.CurrentTracks.tracks[idx].H)
                Pk=Mat_extractROS(self.CurrentTracks.tracks[idx].P)
                SkInv=np.inv(Hk*Pk*Hk.T+self.R_rdr)
                # TODO: Edit for Delphi ESR
                for jdx in range(len(SensorData)):
                    Vc=np.sqrt((self.Vt+SensorData[jdx].vx)^2+SensorData[jdx].vy^2)
                    Beta=  SensorData[jdx].vy_comp/SensorData[jdx].vx_comp# This will be Vx/Vy for delphi esr
                    rho=np.sqrt(SensorData[jdx].pose.x^2+SensorData[jdx].pose.y^2)
                    rhoDot=(SensorData[jdx].pose.x*np.sin(self.psi-Beta)*Vc+SensorData[jdx].pose.y*np.cos(self.psi-Beta)*Vc)/rho
                    y[jdx]=np.array([SensorData[jdx].pose.x,rho,rhoDot])
                    Temp=(y[jdx]-y_est[idx]).T*SkInv*(y[jdx]-y_est[idx])
                    if (Temp<=self.GateThresh^2):
                        SensorOutData[idx].append(SensorData[jdx])
        return SensorOutData

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
        #self.trackManager(self.CamReadings)
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
            self.RdrReadings[-1].vx=data.objects[idx].vx
            self.RdrReadings[-1].vy=data.objects[idx].vy
            self.RdrReadings[-1].vx_comp=data.objects[idx].vx_comp
            self.RdrReadings[-1].vy_comp=data.objects[idx].vy_comp
            self.RdrReadings[-1].stamp=data.header.stamp
        #print(self.RdrReadings[3].stamp)
        self.trackManager(self.RdrReadings)
        


if __name__=='__main__':

	main()
    

            
        
