from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

########################################################################################
import sys, time, traceback, logging
from queue import Queue, SimpleQueue
from threading import Thread
from os.path import abspath, join

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

def startTiming():
    global start_time
    start_time = time.time()


def caculateFPS():
    global frameOutputCount
    global start_time
    if (time.time() - start_time) > 1:
        print("PYSOT_FPS", frameOutputCount / (time.time() - start_time), flush=True)
        frameOutputCount = 0
        start_time = time.time()
    frameOutputCount += 1

def showFrame(video_name, frame, calcFPS=True):
    cv2.imshow(video_name, frame)
    c = cv2.waitKey(10)
    if c == 27:
        jfs.stopTask()
    if calcFPS:
        caculateFPS()

def waitKey():
    c = cv2.waitKey(1)
    if c == 27:
        jfs.stopTask()

def on_mouse(event, x, y, flags, params):
    global isLMouseUp, isLMouseDown, isROISelected, tmp_rect, init_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        tmp_rect[[0, 2]] = x
        tmp_rect[[1, 3]] = y
        isLMouseDown = True
        isLMouseUp = False
    elif event == cv2.EVENT_MOUSEMOVE and isLMouseDown:
        tmp_rect[0] = min(x, tmp_rect[0])
        tmp_rect[1] = min(y, tmp_rect[1])
        tmp_rect[2] = max(x, tmp_rect[0])
        tmp_rect[3] = max(y, tmp_rect[1])
    elif event == cv2.EVENT_LBUTTONUP:
        init_rect = (tmp_rect[0],tmp_rect[1],(tmp_rect[2] -tmp_rect[0]),(tmp_rect[3] - tmp_rect[1]))
        isLMouseDown = False
        isLMouseUp = True

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
                    print(e,flush=True)
                print("Debug - delete queue: delete success",flush=True)  # for debug
            else:
                print("Debug - delete queue: delete not success",flush=True)  # for debug
                time.sleep(0.05)



    def bufferFramesToFrameQueue(self):
        global CurrFrameNum
        Curr_frame_num_of_failure = 0
        num_of_cons_failed_frames = 0
        try:
            while not self.isStopTask:
                    print("Debug - 1",flush=True)
                    imgpath = abspath( join(self.src,str(CurrFrameNum)+".jpg"))
                    print("Debug - 2",flush=True)
                    frame = cv2.imread(imgpath)
                    print("Debug - 3",flush=True)
                    if frame is None:
                        if not (CurrFrameNum == 1):
                             Curr_frame_num_of_failure += 1
                             if Curr_frame_num_of_failure > 5:
                                 CurrFrameNum += 1
                                 print("FrameNum:",str(CurrFrameNum),"reach max failure")
                                 num_of_cons_failed_frames += 1
                                 Curr_frame_num_of_failure = 0
                                 if num_of_cons_failed_frames > 4:
                                    #Stop the program if the program cannot read the last 5 consecutive frames
                                    self.stopTask()
                        print("Debug - cv2.imread() failed. frameNum =", str(CurrFrameNum), flush=True)  # for debug
                        time.sleep(0.1)

                    else:
                        Curr_frame_num_of_failure = 0
                        num_of_cons_failed_frames = 0
                        print("Debug - cv2.imread() succeeded. frameNum =", str(CurrFrameNum), flush=True)  # for debug
                        self.put(frame, CurrFrameNum)
                        self.putToDeleteQueue(imgpath, CurrFrameNum)
                        # time.sleep(0.05) #for debug
                        CurrFrameNum += 1

                    #Video Capture Version: Suffer from sudden read failure when the code is packaged using PyInstaller
                    # isOpened = self.cap.isOpened()
                    # print("cap.isOpened: ",str(isOpened),flush=True)
                    # if isOpened:
                    #     if self.cap.grab():
                    #         print("cap.grab() success", flush=True)
                    #         (ret , frame) = self.cap.retrieve()
                    #         print(CurrFrameNum, str(int(self.cap.get(1))), flush=True) # for debug
                    #         if not ret: #is cap.read() success or not
                    #             print("cap.retrieve() failed. (ret=False) frameNum =",str(CurrFrameNum),flush=True) #for debug
                    #             time.sleep(0.05)
                    #         else:
                    #             print("cap.retrieve() succeed. (ret=True) frameNum =", str(CurrFrameNum),flush=True)  # for debug
                    #             self.put(frame,CurrFrameNum)
                    #             self.putToDeleteQueue(self.src + "/" + str(CurrFrameNum) + ".jpg")
                    #             #time.sleep(0.05) #for debug
                    #             CurrFrameNum += 1
                    #     else:
                    #         print("cap.grab() fails", flush=True)
                    #         time.sleep(0.05)
                    # else:
                    #     self.cap.open(f"{self.src}/%d.jpg",cv2.CAP_IMAGES)

        except (Exception, cv2.error) as e:
            print(str(e),flush=True)


    def put(self,frame,CurrFrameNum):
        if not self.FrameQueue.full():
            self.FrameQueue.put(frame)
            print("Debug - frame queue: put success frameNum =",str(CurrFrameNum), flush=True) #for debug
        else:
            self.FrameQueue.get() #drop frames when the buffer is full
            self.FrameQueue.put(frame)
            print("Debug - frame queue: put and drop success frameNum =",str(CurrFrameNum), flush=True) #for debug

    def read(self):
        while not self.isStopTask:
            #self.printQueueSize()
            if not self.FrameQueue.empty():
                lastestFrame = self.FrameQueue.get()
                yield lastestFrame
                print("Debug - frame queue: read success", flush=True) #for debug
            else:
                time.sleep(0.01)
                waitKey()
                print("Debug - frame queue: read not success", flush=True) #for debug

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
        print("StartReceiveJPGStream", "False", flush=True)
        print("Program stopped - No more frame received (cannot read the last 5 consecutive frames)")
        cv2.destroyAllWindows()
        sys.exit()
        #self.cap.release()

    def printQueueSize(self):
        print("JPS_FrameQueue_Size",self.FrameQueue.qsize(),flush=True) #for debug

    def putToDeleteQueue(self,path,CurrFrameNum):
        self.DeleteQueue.put(path)
        print("Debug - delete queue: put success. frameNum =", str(CurrFrameNum), flush=True)  # for debug


###########################################################################################

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
###########################################################################################
parser.add_argument('--buffer_size', default='5', type=int,
                    help='Real Time Frame Buffer Size')
###########################################################################################

args = parser.parse_args()


# def get_frames(video_name):
#     if not video_name:
#         cap = cv2.VideoCapture(0)
#         # warmup
#         for i in range(5):
#             cap.read()
#         while True:
#             ret, frame = cap.read()
#             if ret:
#                 yield frame
#             else:
#                 break
#     elif video_name.endswith('avi') or \
#             video_name.endswith('mp4'):
#         cap = cv2.VideoCapture(args.video_name)
#         while True:
#             ret, frame = cap.read()
#             if ret:
#                 yield frame
#             else:
#                 break
#     else:
#         images = glob(os.path.join(video_name, '*.jp*'))
#         images = sorted(images,
#          # key=lambda x: int(x.split('')[-1].split('.')[0]))
#         ########################################################################
#          key = lambda x: int(x.replace(args.video_name,"").replace('\\','').split('.')[0]))
#         ########################################################################
#
#         for img in images:
#             frame = cv2.imread(img)
#             yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    #####################################
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    #####################################
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot, map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    #first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'

    #cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    ##########################################
    global jfs
    cv2.namedWindow(video_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(video_name, on_mouse=on_mouse)
    jfs = JPGFileStream(args.video_name,args.buffer_size).startTask()
    print("StartReceiveJPGStream", "True", flush=True)
    InitThread = None #Delcare Thread for Tracker.Init First
    # for frame in get_frames(args.video_name):

    for frame in jfs.read():
    ##########################################
        try:
            # if first_frame:
            #     try:
            #         init_rect = cv2.selectROI(video_name, frame, False, False)
            #     except:
            #         exit()
            #     tracker.init(frame, init_rect)
            #     first_frame = False
            #

            ##########################################################
            if not jfs.isStopTask:
                global isROISelected, isLMouseUp, isLMouseDown, tmp_rect, init_rect,isInitThreadStarted
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
                                print("Frame_Size", str(h), str(w))
                            showFrame(video_name,frame,False)
                    else:
                        isROISelected = not InitThread.isAlive()
                        if isROISelected:
                             startTiming()
                    cv2.rectangle(frame, (int(tmp_rect[0]), int(tmp_rect[1])),
                                  (int(tmp_rect[2]), int(tmp_rect[3])),
                                  (0, 255, 0), 3)
                    showFrame(video_name, frame, False)
                    ##########################################################

                else:

                        outputs = tracker.track(frame)
                        if 'polygon' in outputs:
                            polygon = np.array(outputs['polygon']).astype(np.int32)
                            cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                          True, (0, 255, 0), 3)
                            mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                            mask = mask.astype(np.uint8)
                            mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                            frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                        else:
                            bbox = list(map(int, outputs['bbox']))
                            cv2.rectangle(frame, (bbox[0], bbox[1]),
                                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                          (0, 255, 0), 3)
                            #################################################
                            print("PYSOT_BBOX", str(bbox), flush=True)
                            #################################################
                        # cv2.imshow(video_name, frame)
                        # cv2.waitKey(40)
                        #################################################
                        showFrame(video_name, frame)
                        #################################################

        #cv2.destroyAllWindows()
        except Exception as e:
            print(str(e), flush=True)



if __name__ == '__main__':
    main()
