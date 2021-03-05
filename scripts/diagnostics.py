#!/usr/bin/env python3

import rospy
import dbw_mkz_msgs.msg
import std_msgs
import numpy as np
import time
from sensor_msgs.msg import Image, CompressedImage, Imu, PointCloud2
from radar_msgs.msg import RadarTrackArray

##==================== USAGE:
# Just checks if a topic is being published every 2 second. If not, throws an rospy error


def diagnostic_listener():
    rospy.init_node('diagnostics_observer', anonymous=True)
    while not rospy.is_shutdown():
        try:
            data=rospy.wait_for_message('/flir_boson1/image_rect', Image, timeout=2)
        except:
            rospy.logerr("FLIR CAM1 Not publishing new messages")

        try:
            data=rospy.wait_for_message('/flir_boson2/image_rect', Image, timeout=2)
        except:
            rospy.logerr("FLIR CAM2 Not publishing new messages")

            
        try:
            data=rospy.wait_for_message('/flir_boson3/image_rect', Image, timeout=2)
        except:
            rospy.logerr("FLIR CAM3 Not publishing new messages")

        try:
            data=rospy.wait_for_message('/flir_boson4/image_rect', Image, timeout=2)
        except:
            rospy.logerr("FLIR CAM4 Not publishing new messages")

        try:
            data=rospy.wait_for_message('/flir_boson5/image_rect', Image, timeout=2)
        except:
            rospy.logerr("FLIR CAM5 Not publishing new messages")

        try:
            data=rospy.wait_for_message('/as_tx/radar_tracks', RadarTrackArray, timeout=2)
        except:
            rospy.logerr("RADAR Not publishing new messages")

        try:
            data=rospy.wait_for_message('/vehicle/wheel_speed_report', dbw_mkz_msgs.msg.WheelSpeedReport, timeout=2)
        except:
            rospy.logerr("MKZ Vehicle bus not publishing new messages")

        try:
            data=rospy.wait_for_message('/imu/data', Imu, timeout=2)
        except:
            rospy.logerr("IMU Not publishing new messages")

        try:
            data=rospy.wait_for_message('/velodyne_points', PointCloud2, timeout=2)
        except:
            rospy.logerr("VELODYNE Not publishing new messages")

        try:
            data=rospy.wait_for_message('/zed/zed_node/rgb/image_rect_color/compressed', CompressedImage, timeout=2)
        except:
            rospy.logerr("ZED Not publishing new messages")


                       




if __name__=='__main__':
    try:
       
        diagnostic_listener()
    except rospy.ROSInterruptException:
        pass