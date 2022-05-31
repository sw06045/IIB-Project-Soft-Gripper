#!/usr/bin/env python3
import rospy
from std_msgs.msg import String,  Int32
import time
import serial

class PneumaticControl():
    ## pip install pyserial
    def __init__(self, port=False, ee_port=False, db_host=False):
        self.port = port
        self.ee_port = ee_port
        print (self.ee_port)

        #init pneumatic connection
        if ee_port!=False:
            self.ee = serial.Serial(self.ee_port, 9600)  # open serial port
            while self.ee.isOpen()==False:
                print("Waiting for hand")
            #print("Serial port opened :)")

            self.ee.send_break()
            time.sleep(1) # This is needed to allow MBED to send back command in time!
            ipt = bytes.decode(self.ee.readline())
            print("Connected to",ipt)

    def serial_send(self,cmd,var,wait):
        print("Serial Send Initiated")
        ipt = ""
        self.ee.reset_input_buffer()
        self.ee.write(str.encode(cmd+chr(var+48)+"\n"))
        #wait for cmd acknowledgement
        while True:
            ipt = bytes.decode(self.ee.readline())
            print("gripper data: ", ipt)
            if ipt == "received\r\n":
                break
        #wait for cmd completion
        if wait==True:
            while True:
                ipt = bytes.decode(self.ee.readline())
                print("gripper data: ", ipt)
                if ipt == "done\r\n":
                    print("Completed gripper CMD")
                    break
        return ipt




def callback(data):
    data.data = str(data.data)
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    if int(data.data) == 11:
        print("Suction On")
        Valve.serial_send("S", 1, True)

    elif int(data.data) == 10:
        print("Suction Off")
        Valve.serial_send("S", 0, True)
    
    elif int(data.data) == 21:
        print("SPA On")
        Valve.serial_send("T", 1, True)

    elif int(data.data) == 20:
        print("SPA Off")
        Valve.serial_send("T", 0, True)

    #print(data)




    
def listener(Valve):

    print("initated node")
    rospy.init_node('listener', anonymous=True)
    print("initiate subscriber")
    rospy.Subscriber("chatter", Int32, callback)

    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    Valve = PneumaticControl(ee_port="/dev/ttyACM0")
    listener(Valve)