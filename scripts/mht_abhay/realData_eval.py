

import rospy
import numpy as np
from math import *
from delphi_esr_msgs.msg import EsrEthTx
from radar_msgs.msg import RadarTrackArray
from nuscenes2bag.msg import RadarObjects
from derived_object_msgs.msg import ObjectWithCovarianceArray
from mwis import graph
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from darknet_ros_msgs.msg import BoundingBoxes
from dbw_mkz_msgs.msg import SteeringReport
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud2
import time
import csv


def main():

    rospy.init_node("Sim_MHT")

    process = mht()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


class mht():

    def __init__(self):
        self.max_track = 20
        self.pruning_window = 5
        self.n = 10			# for filter1
        self.dth = 10
        self.filter1Ther = 2
        self.V = 3000/1		# scan area of radar
        self.Pd = 0.9
        self.min_score = 20
        self.min_miss = 10

        self.track_init = []
        self.tracks = dict()
        self.filter1_tracks = []
        self.dispTracks = []
        self.frame = 1
        self.P0 = np.matrix(np.eye(2))
        self.Qd = 0.00001*np.matrix(np.eye(2))
        self.R = 1*np.array([[0.25, 0], [0, 0.01]])
        self.speed = 0

        self.missed_Detection = 1*float(np.log(1 - self.Pd))

        self.bridge = CvBridge()
        self.detections = []
        self.BBox, self.BBdetections = [], []
        self.updated_id = []
        self.frame_data = []
        self.BBflag = 0

        filePathPrefix=str("/home/vamsi/Tracking/py-motmetrics/motmetrics/res_dir/")
        self.delta_x = 0
        self.delta_y = 0 # Assuming that the radar and camera are on same centerline
        self.delta_z = 1.0414/2
        self.H_FOV=190
        self.V_FOV=41 #Calculated based on aspect ratio
        self.HorzOffset=0 # Manual horizontal (Y-direction) offset for radar in pixels
        self.VertOffset=-30 # Manual vertical (Z-direction) offset for radar in pixels
        self.DestF=open((filePathPrefix+'seq1'+'.txt'),"w")
        self.ImageExists=0
        self.BBheight=90
        self.BBWidth=90 # For now, static
        self.FrameInit=1

        self.Vertical_correction = 4

        self.image_pub = rospy.Publisher("fused_image3", Image, queue_size=10)

        rospy.loginfo('Published fused image')

        rospy.Subscriber("/as_tx/objects",
                         ObjectWithCovarianceArray, self.radar)
        # rospy.Subscriber('/vehicle/twist',
        #                  SteeringReport, self.steering_report)
        rospy.Subscriber("/darknet_ros/bounding_boxes",
                         BoundingBoxes, self.bounding_boxes)
        rospy.Subscriber('/Thermal_Panorama', Image, self.image)
        rospy.Subscriber('/os_cloud_node/points', PointCloud2,self.writeToFile) #Only write to file everytime a new lidar PCL is published


    class detection:
        def __init__(self, detection, BB, _id):
            self.id = _id
            self.x = detection.pose.pose.position.x
            self.y = detection.pose.pose.position.y
            self.vx = detection.twist.twist.linear.x
            self.vy = detection.twist.twist.linear.y
            self.BB = BB


    class track:
        def __init__(self, x, y, vx, vy, P, score, BB, miss, y_disp, tree_id):
            self.x = x
            self.y = y
            self.vx = vx
            self.vy = vy
            self.P = P
            self.score = score
            self.BB = BB
            self.miss = miss
            self.y_disp = y_disp
            self.tree_id = tree_id


    def steering_report(self, data):
        self.speed = int(data.speed)
        print('speed',self.speed)


    def image(self, data):
        self.RawImage=self.bridge.imgmsg_to_cv2(data, "rgb8")
        self.ImageExists=1
        self.writeToFile()


    def writeToFile(self,dataDump):
        frame=self.FrameInit
        self.FrameInit+=1
        # print(frame)

        for index,track in self.tracks.items():
            #write to file
            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            
            _id=track.tree_id
            RadarX=track.x+self.delta_x
            RadarY=track.y
            RadarZ=0.0+self.delta_z
            RadarAnglesH=-np.degrees(np.arctan(np.divide(RadarY,RadarX)))
            RadarAnglesV=np.abs(np.degrees(np.arctan(np.divide(RadarZ,RadarX)))+self.Vertical_correction) #will always be negative, so correct for it
            
            
            if self.ImageExists==1:
                imageTemp = self.RawImage
                CameraX=RadarAnglesH*(self.RawImage.shape[1]/self.H_FOV) + self.RawImage.shape[1]/2 +self.HorzOffset# Number of pixels per degree,adjusted for shifting origin from centerline to top left
                CameraY=RadarAnglesV*(self.RawImage.shape[0]/self.V_FOV) +256 +self.VertOffset -RadarX*np.sin(np.radians(4)) # Number of pixels per degree,adjusted for shifting origin from centerline to top left

               
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
                outLine=str(frame)+' '+str(_id)+' '+str(bb_left)+' '+str(bb_top)+' '+str(bb_width)+' '+str(bb_height)+' '+str(conf)+' '+str(x)+' '+str(y)+' '+str(z)+'\n'
                self.DestF.write(outLine)


    def bounding_boxes(self, data):
        self.BBox = []

        for i in range(len(data.bounding_boxes)):
            if data.bounding_boxes[i].Class == "car":
                self.BBox.append((data.bounding_boxes[i].xmin, data.bounding_boxes[i].ymin, data.bounding_boxes[i].xmax, data.bounding_boxes[i].ymax,
                              data.bounding_boxes[i].probability))


    def BBfilter(self,BBox):
        print(len(BBox))
        for detection in self.detections:
            RadarX=detection.x+self.delta_x
            RadarY=detection.y
            RadarZ=0.0+self.delta_z
            if self.ImageExists==1:
                RadarAnglesH=-np.degrees(np.arctan(np.divide(RadarY,RadarX)))
                RadarAnglesV=np.abs(np.degrees(np.arctan(np.divide(RadarZ,RadarX)))+self.Vertical_correction) #will always be negative, so correct for it
                CameraX=RadarAnglesH*(self.RawImage.shape[1]/self.H_FOV) + self.RawImage.shape[1]/2 +self.HorzOffset# Number of pixels per degree,adjusted for shifting origin from centerline to top left
                CameraY=RadarAnglesV*(self.RawImage.shape[0]/self.V_FOV) + 256 +self.VertOffset -RadarX*np.sin(np.radians(4)) # Number of pixels per degree,adjusted for shifting origin from centerline to top left
                for j, BB in enumerate(BBox):
                    if BB[0] <= CameraX <= BB[2] and BB[1] <= CameraY <= BB[3]:
                        detection.BB = j

        for track in self.tracks.values():
            RadarX=track.x+self.delta_x
            RadarY=track.y
            RadarZ=0.0+self.delta_z
            if self.ImageExists==1:
                RadarAnglesH=-np.degrees(np.arctan(np.divide(RadarY,RadarX)))
                RadarAnglesV=np.abs(np.degrees(np.arctan(np.divide(RadarZ,RadarX)))+self.Vertical_correction) #will always be negative, so correct for it
                CameraX=RadarAnglesH*(self.RawImage.shape[1]/self.H_FOV) + self.RawImage.shape[1]/2 +self.HorzOffset# Number of pixels per degree,adjusted for shifting origin from centerline to top left
                CameraY=RadarAnglesV*(self.RawImage.shape[0]/self.V_FOV) +256 +self.VertOffset -RadarX*np.sin(np.radians(4)) # Number of pixels per degree,adjusted for shifting origin from centerline to top left
                for j, BB in enumerate(BBox):
                    if BB[0] <= CameraX <= BB[2] and BB[1] <= CameraY <= BB[3]:
                        track.BB = j


    def radar(self, data):
        self.detections = []
        self.filter1 = []
        self.updated_id = []
        BBox = self.BBox
        

        ############## Track initialization filter 1 #######################################################

        for i in range(len(data.objects)):
            if data.objects[i].pose.pose.position.x == 0 and data.objects[i].pose.pose.position.y ==0: 
                pass
            else:
                self.detections.append(self.detection(data.objects[i], -1, i)) 

                # -1 default value for BB, id is just sequence number for detection
        # for detection in self.detections:
        #     print(detection.y)

        ################################ Filter1 #######################################################
        flag = [0]*len(self.detections)     #
        del_index = []                      #

        for i in range(len(self.track_init)):
            
            X = np.array([[self.track_init[i][0].x], [self.track_init[i][0].y]])
            delX = 0.02*np.array([[self.track_init[i][0].vx], [self.track_init[i][0].vy]])
            Xhat = X + delX    
            # print('XHat',Xhat)                                                                                     #                                                                                    # Kalman Prediction steps
            f2 = 0
            for detection in self.detections:
                Y = np.array([[detection.x], [detection.y]])
                # print('Y',Y)
                # print((Xhat-Y).shape)
                dist = np.linalg.norm(Xhat-Y)  #(Xhat-Y).transpose()*(Xhat-Y)
                # print('dist',dist)
                if dist<self.filter1Ther:                                                                 # There is a possibility that many detection satisfy this ""NEED TO ADDRESS THIS""
                    self.track_init[i][0] = detection
                    self.track_init[i][1] = min(self.track_init[i][1] + 1, self.n+3)
                    flag[self.detections.index(detection)] = 1
                    f2 = 1

            if f2 == 0:
                self.track_init[i][1] -= 1
                if self.track_init[i][1] <= self.n -2:
                    del_index.append(i)

            if self.track_init[i][1] >= self.n and self.track_init[i][0] not in self.filter1 :          #so that we dont get duplicate track_ids
                self.filter1.append(self.track_init[i][0])

        for j in range(len(del_index)):                                                                            # delete all track_init whose score drops below threshold
            del self.track_init[del_index[j]-j]

        for i in range(len(flag)):
            if flag[i] == 0:
                self.track_init.append([self.detections[i],1])
        ####################################################################################################


        self.BBfilter(BBox)

        if len(self.tracks) != 0:
            self.update_tracks()

        print('filter1',len(self.filter1))
        print('tracks',len(self.tracks))
        for detection in self.detections:
            if detection in self.filter1:# and detection.id not in self.updated_id:
                # print('speed',self.speed)
                # if self.speed>10:
                #     print('speed',self.speed)
                #     detection_speed = np.sqrt(np.square(detection.vx) + np.square(detection.vy))

                #     if self.speed-2 <= detection_speed <= self.speed+2:
                #         continue
                #     else:
                if detection.BB != -1:
                    updatescore = 10 + self.BBox[detection.BB][4]*10
                    y_disp = int(
                        0.5*(self.BBox[detection.BB][1]+self.BBox[detection.BB][3]))
                # else:
                #     updatescore = 10
                #     y_disp = 0

                    track_id = [detection.id]
                    self.tracks[tuple(track_id)] = self.track(
                        detection.x, detection.y, detection.vx, detection.vy, self.P0, updatescore, detection.BB, 0, y_disp,0)
        

        self.pruning()
        self.frame += 1

        print('################################  new frame  ######################################')

    def update_tracks(self):

        # prediction step
        index = self.tracks.keys()
        track = self.tracks.values()

        updated_tracks = []

        for i in range(len(index)):

            del self.tracks[index[i]]

            flag = 0
            tree_id = track[i].tree_id

            X = np.array([[track[i].x], [track[i].y]])
            P = track[i].P
            delX = 0.02*np.array([[track[i].vx], [track[i].vy]])

            Xhat = X + delX                                                                                         #
            Phat = P + self.Qd                                                                                      # Kalman Prediction steps
            kalman_gain = Phat*np.linalg.inv(Phat+self.R)                                                           #

############## New possible tracks from observations inside track gate ############################

            for detection in self.detections:
                Y = np.array([[detection.x], [detection.y]])
                dist = self.MH_distance(Y, Xhat, Phat)

                if dist <= self.dth:                                                                            # Gating
                    flag = 1
                    self.updated_id.append(detection.id)
                    updatescore = float((np.log(self.V/(2*np.pi)) - 0.5 * np.log(np.linalg.det(Phat)) - dist/2))
                    y_disp = track[i].y_disp

                    if detection.BB != -1:
                        updatescore = updatescore + self.BBox[detection.BB][4]*10
                        y_disp = int(0.5*(self.BBox[detection.BB][1]+self.BBox[detection.BB][3]))

                    err = Y-X                                                                                       #
                    Xnew = Xhat + kalman_gain*err                                                                   # Kalman Update steps
                    Pnew = [np.matrix(np.eye(2)) - kalman_gain] * Phat                                              #

                    track_id = list(index[i])
                    if len(track_id) >= self.pruning_window:
                        track_id.pop(0)
                    track_id.append(detection.id)
                    score = track[i].score + updatescore

                    updated_tracks.append((tuple(track_id), self.track(float(Xnew[0][0]), float(
                        Xnew[1][0]), detection.vx, detection.vy, Pnew, score, detection.BB, 0, y_disp,tree_id)))


            if flag == 0:
                updatescore = self.missed_Detection
                track_id = list(index[i])
                last_id = track_id[-1]

                if len(track_id) >= self.pruning_window:
                    track_id.pop(0)
                track_id.append(-1)
                miss = track_id.count(-1)                           #count of -1 in track.id 
                score = track[i].score + miss*miss*updatescore      #to decrease the score rapidly

                track[i].miss += 1

                if track[i].miss < self.min_miss and score>=self.min_score:
                    track[i].score = score
                    updated_tracks.append((tuple(track_id), track[i]))

        for track in updated_tracks:
            self.tracks[track[0]] = track[1]


    def MH_distance(self, Y, X, P):
        mh_dist = (X-Y).transpose()*np.linalg.inv(P)*(X-Y)
        return mh_dist


    def pruning(self):
        conflicting_tracks = []
        index = self.tracks.keys()
        track = self.tracks.values()
        n = len(index)
        # print('pruning',n)

############## Edges of the graph #################################################################
        for i in range(n):
            BB1 = track[i].BB
            for j in range(i+1, len(index)):
                if track[j].BB == BB1 and BB1 != -1:
                    conflicting_tracks.append((i, j))
                else:
                    id1 = list(index[i])
                    id2 = list(index[j])

                    for k in range(1, min(len(id1), len(id2))+1):
                        if id1[-k] != -1 and id1[-k] == id2[-k]:
                            conflicting_tracks.append((i, j))

        adj_mat = np.ones((n, n), dtype=int)  # Create the NxN matrix
        np.fill_diagonal(adj_mat, 0)
        for edge in conflicting_tracks:
            (a, b) = edge
            adj_mat[a, b] = 0
            adj_mat[b, a] = 0

        gra = graph()
        ind_sets = gra.mwis(adj_mat)
        max_weight = 0
        mwis = []
        for ind_set in ind_sets:
            set_weight = sum([track[i].score for i in ind_set])
            if set_weight > max_weight:
                max_weight = set_weight
                mwis = ind_set
        # print('mwis',mwis)
        mwis_score = []
        for i in mwis:
            mwis_score.append((i, track[i].score))

        top_score = sorted(mwis_score, key=lambda x: x[1], reverse=True)[
            :min(self.max_track, len(mwis))]

        top_ids = [x[0] for x in top_score]

        free_trees = list(range(1,self.max_track+1))
        for _id in top_ids:
            if self.tracks[index[_id]].tree_id != 0:
                free_trees.remove(self.tracks[index[_id]].tree_id)

        for _id in top_ids:
            if self.tracks[index[_id]].tree_id == 0:
                self.tracks[index[_id]].tree_id = free_trees.pop(0)
 
        self.dispTracks = []

        # to average out the disp_track for multiple track family tree

        # top_idBB = [track[i].BB for i in top_ids]
        # for i in range(len(top_idBB)):
        #   BB = top_idBB[i]
        #   x,y = [],[]
        #   for j in range(n):
        #       if track[j].BB == BB:
        #           x.append(track[j].x)
        #           y.append(track[j].y)
        #   X,Y = sum(x)/len(x),sum(y)/len(y)
        #   self.dispTracks.append((X,Y))

        # top_idBB =[]

        for i in range(n):

            if i in top_ids:
                # top_idBB.append(track[i].BB)
                self.dispTracks.append(
                    (track[i].x, track[i].y, track[i].y_disp))
                continue
            else:
                flag = 0
                for j in top_ids:
                    id1, id2 = index[i], index[j]
                    for k in range(1, min(len(id1), len(id2))):
                        if id1[-k] != -1 and id1[-k] == id2[-k]:
                            flag = 1
                            self.tracks[index[i]].tree_id = self.tracks[index[j]].tree_id 

                if flag == 0:
                    del self.tracks[index[i]]





if __name__ == '__main__':

    main()