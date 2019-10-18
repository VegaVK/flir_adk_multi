
#!/usr/bin/env python
import rospy
import std_msgs
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import numpy as np
#### BUILT FOR 2 CAMERAS CURRENTLY
def main():
    rospy.init_node('Some_publisher', anonymous=True)
    vsInst=vid_stitch()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

class vid_stitch:

    def __init__(self):
        self.bridge=CvBridge()
        self.image_pub = rospy.Publisher("stitched_image2",Image,queue_size=100)
        self.image1=np.array([1])
        self.image2=self.image1
        self.overlapPix=180# number of pixels (horizontal) to average over

        
        rospy.Subscriber('/flir_boson3/image_rect', Image, self.buildimage1)
        rospy.Subscriber('/flir_boson4/image_rect', Image, self.buildimage2)

    
    def buildimage1(self,data):
        self.image1=self.bridge.imgmsg_to_cv2(data, "mono8")
    def buildimage2(self,data):
        self.image2=self.bridge.imgmsg_to_cv2(data, "mono8")
        self.stitchfun() # Put this in the last buildimage() callback

    def stitchfun(self):
        # For two cameras, basic stitching, average over common pixels
        sift = cv2.xfeatures2d.SIFT_create()
        # find key points
        kp1, des1 = sift.detectAndCompute(self.image2,None)
        kp2, des2 = sift.detectAndCompute(self.image1,None)#cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_,kp1,None))#FLANN_INDEX_KDTREE = 0
        #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        #search_params = dict(checks = 50)
        #match = cv2.FlannBasedMatcher(index_params, search_params)
        match = cv2.BFMatcher()
        matches = match.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.03*n.distance:
                good.append(m)
        draw_params = dict(matchColor=(0,255,0),singlePointColor=None,flags=2)
        self.imageComm = cv2.drawMatches(self.image2,kp1,self.image1,kp2,good,None,**draw_params)
        #cv2.imshow("original_image_drawMatches.jpg", img3)MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            M=np.array([1,0,0;0,1,0;0,0,1])
            h,w = self.image2.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, M)
            self.image1 = cv2.polylines(self.image1,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            #cv2.imshow("original_image_overlapping.jpg", img2)
        else:
            print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))
        self.Final = cv2.warpPerspective(self.image2,M,(self.image1.shape[1] + self.image2.shape[1], self.image1.shape[0]))
        self.Final[0:self.image1.shape[0],0:self.image1.shape[1]] = self.image1
        #cv2.imshow("original_image_stitched.jpg", self.Final)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.Final, "mono8"))
        rospy.loginfo('Published stitched image')
        
    def trim(frame):
            #crop top
            if not np.sum(frame[0]):
                return trim(frame[1:])
            #crop top
            if not np.sum(frame[-1]):
                return trim(frame[:-2])
            #crop top
            if not np.sum(frame[:,0]):
                return trim(frame[:,1:])
            #crop top
            if not np.sum(frame[:,-1]):
                return trim(frame[:,:-2])
            return framecv2.imshow("original_image_stitched_crop.jpg", trim(dst))
        #cv2.imsave("original_image_stitched_crop.jpg", trim(dst))
        


if __name__=='__main__':
    main()
#######################

