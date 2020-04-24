###########################################################################################

# Real-time Object Tracking Pipeline (ROTP)

###########################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import cv2
import torch
import numpy as np

import sys, time
from queue import Queue, SimpleQueue
from threading import Thread
from os.path import abspath, join
from statistics import mean


###########################################################################################
#Import the tracking function here

from PySOT import objecttracking
from DBLLNet import llenhancement, Utils

###########################################################################################

###########################################################################################
#Set the required arg here

parser = argparse.ArgumentParser(description='object tracking and low light enhancement module')
parser.add_argument('--pysotconfig', type=str, help='pysot config file')
parser.add_argument('--pysotcheckpoint', type=str, help='pysot tracking model\'s checkpoint path')
parser.add_argument('--dbllnetcheckpoint', type=str, help='DBLLNet model\'s checkpoint path')
parser.add_argument('--data', default='', type=str,help='DJI data path')
parser.add_argument('--buffer_size', default='5', type=int, help='Real Time Frame Buffer Size')
parser.add_argument('--dark_mode', dest='dark_mode', action='store_true')
parser.add_argument('--no-dark_mode', dest='dark_mode', action='store_false')
parser.add_argument('--synthetic', dest='synthetic', action='store_true')
parser.add_argument('--no-synthetic', dest='synthetic', action='store_false')
parser.add_argument('--enhancement_mode', dest='enhancement_mode', type=str, default="DBLLNet")
parser.set_defaults(dark_mode=False)
args = parser.parse_args()

###########################################################################################



###########################################################################################
#   GLOBAL VARIABLES FOR PROCESSING REAL TIME FRAMES INPUT AND THE SELECT ROI ACTION

start_time = 0
frameOutputCount = 0
isLMouseDown = False
isLMouseUp = False
isROISelected = False
tmp_rect = np.zeros(4)
init_rect = None
jfs = None
isInitThreadStarted = False
CurrFrameNum = 1
FPS = []

###########################################################################################
#   UTILS FUNCTIONS

def startTiming():
    global start_time
    start_time = time.time()


def caculateFPS():
    global frameOutputCount
    global start_time
    global FPS
    if (time.time() - start_time) > 1:
        fps =  frameOutputCount / (time.time() - start_time)
        print("\nPYSOT_FPS {}".format(fps), flush=True)
        FPS.append(fps)
        frameOutputCount = 0
        start_time = time.time()
    frameOutputCount += 1

def showFrame(data, frame, calcFPS=True):
    cv2.imshow(data, frame)
    c = cv2.waitKey(1)
    if c == 27:
        jfs.stopTask()
    if calcFPS:
        caculateFPS()

def waitKey():
    c = cv2.waitKey(1)
    if c == 27:
        jfs.stopTask()

###########################################################################################
#   FUNCTION FOR PROCESSING SELECT ROI ACTION BY USER IN REAL-TIME

def on_mouse(event, x, y,flag, data):
    global isLMouseUp, isLMouseDown, isROISelected, tmp_rect, init_rect

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONUP:
        if event == cv2.EVENT_LBUTTONDOWN:
            tmp_rect[0] = x
            tmp_rect[1] = y
            tmp_rect[2] = x
            tmp_rect[3] = y
            isLMouseDown = True
        else:
            init_rect = (tmp_rect[0],tmp_rect[1],(tmp_rect[2] - tmp_rect[0]),(tmp_rect[3] - tmp_rect[1]))
            isLMouseDown = False

        isLMouseUp = not isLMouseDown

    elif event == cv2.EVENT_MOUSEMOVE:
        if isLMouseDown:
            tmp_rect[2] = x
            tmp_rect[3] = y


###########################################################################################
# A CLASS TO PROCESS IMAGE FRAMES INPUT IN REAL-TIME

class JPGFileStream:
    def __init__(self, src, qSize):
        self.src = src
        #self.cap = cv2.VideoCapture(f"{src}/%d.jpg", cv2.CAP_IMAGES)
        self.isStopTask = False
        self.FrameQueue = Queue(maxsize=qSize)
        self.DeleteQueue = SimpleQueue()
        self.t = None
        self.t1 = None

    def deleteFrames(self):
        while not self.isStopTask:
            if not self.DeleteQueue.empty():
                tmpToBeDeletedFramePath = self.DeleteQueue.get()
                try:
                    time.sleep(0.02)
                    os.remove(tmpToBeDeletedFramePath)
                except (PermissionError,BlockingIOError,Exception) as e:
                    z = e
                    print("\n",e,flush=True)
                print("\nDebug - delete queue: delete success",flush=True)  # for debug
            else:
                print("\nDebug - delete queue: delete not success",flush=True)  # for debug
                time.sleep(0.05)

    def bufferFramesToFrameQueue(self):
        global CurrFrameNum
        Curr_frame_num_of_failure = 0
        num_of_cons_failed_frames = 0
        try:
            while not self.isStopTask:
                    print("\nDebug - 1",flush=True)
                    imgpath = abspath( join(self.src,f'{CurrFrameNum}'+".jpg"))
                    print("\nDebug - 2",flush=True)
                    frame = cv2.imread(imgpath)
                    print("\nDebug - 3",flush=True)
                    if frame is None:
                        if not (CurrFrameNum == 1):
                             Curr_frame_num_of_failure += 1
                             if Curr_frame_num_of_failure > 5:
                                 CurrFrameNum += 1
                                 print("\nFrameNum:",str(CurrFrameNum),"reach max failure",flush=True)
                                 num_of_cons_failed_frames += 1
                                 Curr_frame_num_of_failure = 0
                                 if num_of_cons_failed_frames > 4:
                                    #Stop the program if the program cannot read the last 5 consecutive frames
                                    self.stopTask()
                        print("\nDebug - cv2.imread() failed. frameNum = " + str(CurrFrameNum), flush=True)  # for debug
                        time.sleep(0.1)

                    else:
                        Curr_frame_num_of_failure = 0
                        num_of_cons_failed_frames = 0
                        print("\nDebug - cv2.imread() succeeded. frameNum = "+ str(CurrFrameNum), flush=True)  # for debug
                        self.put(frame, CurrFrameNum)
                        self.putToDeleteQueue(imgpath, CurrFrameNum)
                        # time.sleep(0.05) #for debug
                        CurrFrameNum += 1

                    #Video Capture Version:
                    # isOpened = self.cap.isOpened()
                    # print("cap.isOpened: ",str(isOpened),'\n',flush=True)
                    # if isOpened:
                    #     if self.cap.grab():
                    #         print("cap.grab() success",'\n', flush=True)
                    #         (ret , frame) = self.cap.retrieve()
                    #         print(CurrFrameNum, str(int(self.cap.get(1))),'\n', flush=True) # for debug
                    #         if not ret: #is cap.read() success or not
                    #             print("cap.retrieve() failed. (ret=False) frameNum =",str(CurrFrameNum),'\n',f lush=True) #for debug
                    #             time.sleep(0.05)
                    #         else:
                    #             print("cap.retrieve() succeed. (ret=True) frameNum =", str(CurrFrameNum),'\n', flush=True)  # for debug
                    #             self.put(frame,CurrFrameNum)
                    #             self.putToDeleteQueue(self.src + "/" + str(CurrFrameNum) + ".jpg")
                    #             #time.sleep(0.05) #for debug
                    #             CurrFrameNum += 1
                    #     else:
                    #         print("cap.grab() fails", '\n', flush=True)
                    #         time.sleep(0.05)
                    # else:
                    #     self.cap.open(f"{self.src}/%d.jpg",cv2.CAP_IMAGES)

        except (Exception, cv2.error) as e:
            print("\n"+str(e),flush=True)


    def put(self,frame,CurrFrameNum):
        if not self.FrameQueue.full():
            self.FrameQueue.put(frame)
            print("\nDebug - frame queue: put success frameNum = " + str(CurrFrameNum), flush=True) #for debug
        else:
            self.FrameQueue.get() #drop frames when the buffer is full
            self.FrameQueue.put(frame)

            print("\nDebug - frame queue: put and drop success frameNum = " + str(CurrFrameNum), flush=True) #for debug

    def read(self):
        while not self.isStopTask:
            #self.printQueueSize()
            if not self.FrameQueue.empty():
                lastestFrame = self.FrameQueue.get()
                yield lastestFrame
                print("\nDebug - frame queue: read success", flush=True) #for debug
            else:
                time.sleep(0.01)
                waitKey()
                print("\nDebug - frame queue: read not success", flush=True) #for debug

    def startTask(self):
        self.t = Thread(target=self.bufferFramesToFrameQueue)
        self.t.daemon = True
        self.t.start()
        self.t1 = Thread(target=self.deleteFrames)
        self.t1.daemon = True
        self.t1.start()
        return self

    def stopTask(self):
        # Stop the program if the program cannot read the last 5 consecutive frames
        self.isStopTask = True
        print("\nStartReceiveJPGStream False", flush=True)
        print("\nProgram stopped - No more frame received (cannot read the last 5 consecutive frames)",flush=True)
        if len(FPS) != 0:
            print("Average FPS: {}".format(mean(FPS)))
        cv2.destroyAllWindows()
        sys.exit()
        #self.cap.release()

    def printQueueSize(self):
        print("\nJPS_FrameQueue_Size {}".format(self.FrameQueue.qsize()),flush=True) #for debug

    def putToDeleteQueue(self,path,CurrFrameNum):
        self.DeleteQueue.put(path)
        print("\nDebug - delete queue: put success. frameNum = "+ str(CurrFrameNum),flush=True)  # for debug

###########################################################################################
# MAIN PROGRAM

def main():

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True

    ###########################################################################################
    #Main: init the tracker here

    tracker = objecttracking.objtracking_init(checkpt=args.pysotcheckpoint, config=args.pysotconfig, cuda = cuda, device=device)
    if args.dark_mode:
        enhancer = llenhancement.llenhance_init(checkpoint=args.dbllnetcheckpoint, device = device,cm=1)


    ###########################################################################################

    if args.data:
        data = args.data.split('/')[-1].split('.')[0]

    global jfs
    cv2.namedWindow(data, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(data, on_mouse=on_mouse)
    jfs = JPGFileStream(args.data,args.buffer_size).startTask()
    # A signal sent to UWPDTP to start collecting frames from drone in real-time
    print("\nStartReceiveJPGStream True", flush=True)
    InitThread = None #Delcare Thread for Tracker.Init First

    for frame in jfs.read():
        try:
            ###########################################################################################
            if args.synthetic:#FOR DEMO USE
                frame = Utils.decrease_brightness(frame) #FOR DEMO USE

            # Place the enhancement method (function) here

            if args.dark_mode:
                frame = llenhancement.enhance(enhancer, frame, device,mode=args.enhancement_mode)

            ###########################################################################################

            if not jfs.isStopTask:
                global isROISelected, isLMouseUp, isLMouseDown, tmp_rect, init_rect, isInitThreadStarted
                if not isROISelected:
                    if not isInitThreadStarted:
                            if isLMouseDown:
                                cv2.rectangle(frame, (int(tmp_rect[0]), int(tmp_rect[1])),
                                              (int(tmp_rect[2]), int(tmp_rect[3])),
                                              (255, 255, 255), 3)
                            elif isLMouseUp:
                                InitThread = Thread(target=tracker.init,args=[frame,init_rect])
                                InitThread.daemon = True
                                InitThread.start()
                                isInitThreadStarted = True
                                cv2.rectangle(frame, (int(tmp_rect[0]), int(tmp_rect[1])),
                                              (int(tmp_rect[2]), int(tmp_rect[3])),
                                              (0, 255, 0), 3)
                                (h, w) = frame.shape[:2]
                                print("\nFrame_Size" + " " + str(h)+ " " + str(w), flush=True)
                            showFrame(data,frame)
                    else:
                        isROISelected = not InitThread.isAlive()
                        if isROISelected:
                             startTiming()
                    cv2.rectangle(frame, (int(tmp_rect[0]), int(tmp_rect[1])),
                                  (int(tmp_rect[2]), int(tmp_rect[3])),
                                  (0, 255, 0), 3)
                    showFrame(data, frame)
                else:
################################################################################
#Place the tracking method (function) here
                        str_bbox, frame = objecttracking.track(tracker,frame)
                        print("\nPYSOT_BBOX" +" "+ str_bbox, flush=True)
                        showFrame(data, frame)
#################################################################################
        except Exception as e:
            print("\n"+str(e), flush=True)




if __name__ == '__main__':
    main()

#Python Main.py --pysotconfig C:/Users/Family/DTP_Data/Source/PySOT/experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml --pysotcheckpoint C:/Users/Family/DTP_Data/Source/PySOT/experiments/siamrpn_alex_l234_dwxcorr/model.pth --data C:/Users/Family/DTP_Data/DJI/Output --buffer_size 5 --no-dark_mode --dbllnetcheckpoint C:/Users/Family/DTP_Data/Source/DBLLNet/checkpoint/model.pth