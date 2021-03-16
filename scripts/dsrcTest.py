#!/usr/bin/env python
import rospy
import numpy as np
import std_msgs
from  flir_adk_multi.msg import pingStatus
from  flir_adk_multi.msg import pingLatency

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
    command = ['ping', param, '1', '-W', '1', host] # Timeout in seconds

    return subprocess.call(command) == 0

def main():
    host='192.168.100.3'
    rospy.init_node('dsrcTest', anonymous=True)
    statusPub = rospy.Publisher("ping_status",pingStatus ,queue_size=100)
    latencyPub = rospy.Publisher("ping_ms",pingLatency ,queue_size=100)
    r = rospy.Rate(10) # 10hz
    statusMsg=pingStatus()
    latencyMsg=pingLatency()
    while not rospy.is_shutdown():
        pingStatusVal=ping(host)
        statusMsg.header.stamp = rospy.Time.now() 
        statusMsg.status.data=float(pingStatusVal)
        statusPub.publish(statusMsg)
        latencyMsg.header.stamp = rospy.Time.now() 
        if pingStatusVal==1.0:
            pingDelay = subprocess.Popen(["ping", "-c", "1", "-W", "1" ,host],stdout = subprocess.PIPE, stderr=subprocess.PIPE) # Timeout in seconds
            out = str(pingDelay.communicate())
            # print(out)
            matcher = re.compile("time=(\d+.\d+) ms")
            try:
                matcherOutput=(matcher.search(out, re.MULTILINE).groups())
                # print(float(matcherOutput[0]))
                latencyMsg.latency.data=float(matcherOutput[0])
                latencyPub.publish(latencyMsg)
            except:
                pass
        else:
            rospy.logwarn('Lost Connection')
        r.sleep()
if __name__=='__main__':
    main()
#######################
