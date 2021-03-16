#!/usr/bin/env python
import rospy
import numpy as np
import std_msgs
import sys
import os
import platform    # For getting the operating system name
import subprocess  # For executing a shell command
import re




# host= ['1.1.1.1']
# temp=os.system('ping ' + host[0])
# print(temp)

def ping(host):
    """
    Returns True if host (str) responds to a ping request.
    Remember that a host may not respond to a ping (ICMP) request even if the host name is valid.
    """

    # Option for the number of packets as a function of
    param = '-n' if platform.system().lower()=='windows' else '-c'

    # Building the command. Ex: "ping -c 1 google.com"
    command = ['ping', param, '1', '-w', '300', host]

    return subprocess.call(command) == 0

def main():
    host='192.168.100.3'
    rospy.init_node('dsrcTest', anonymous=True)
    pingPub = rospy.Publisher("ping_status",std_msgs.msg.Float32 ,queue_size=100)
    latencyPub = rospy.Publisher("ping_ms",std_msgs.msg.Float32 ,queue_size=100)

    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        pingStatus=ping(host)
        if pingStatus==1.0:
            pingPub.publish(float(pingStatus))
            pingDelay = subprocess.Popen(["ping", "-c", "1", "-w", "100" ,host],stdout = subprocess.PIPE, stderr=subprocess.PIPE)
            out = str(pingDelay.communicate())
            # print(out)
            matcher = re.compile("time=(\d+.\d+) ms")
            latency=[]
            try:
                matcherOutput=(matcher.search(out, re.MULTILINE).groups())
                latency=float(matcherOutput[0])
            # print(latency)
            # TODO: CODE IS BROKEN - doesnt publish zero ping_status when connection is broken.....
            except:
                rospy.loginfo_once('Lost Connection Midway')
            if latency!=[]:
                latencyPub.publish(latency)
        else:
            pingPub.publish(float(pingStatus))
            rospy.loginfo('Lost Connection')
        r.sleep()
if __name__=='__main__':
    main()
#######################
