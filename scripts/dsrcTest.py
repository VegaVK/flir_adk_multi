#!/usr/bin/env python
import rospy
import numpy as np
import std_msgs
import sys
import os
import platform    # For getting the operating system name
import subprocess  # For executing a shell command


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
    pingPub = rospy.Publisher("ping_mov_avg",std_msgs.msg.Float32 ,queue_size=100)
    r = rospy.Rate(10) # 10hz
    pingBuffer=[]
    pingfile=open('pingTestFile.txt','a')
    while not rospy.is_shutdown():
        pingOut=ping(host)
        pingfile.write(str(int(pingOut)) + '\n')
        if len(pingBuffer)<20:
            pingBuffer.append(pingOut)
        else:
            pingBuffer.pop(0)
            pingBuffer.append(pingOut)
        movingAverage=np.mean(pingBuffer)*100.0
        pingPub.publish(movingAverage)
        r.sleep()
    pingfile.close()
if __name__=='__main__':
    main()
#######################
