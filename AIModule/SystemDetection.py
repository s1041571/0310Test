import cv2
from threading import Thread
import pygame
import sys
from datetime import datetime
sys.path.append('./AIModule')
from utils.plots import plot_fail_touch
import numpy as np
import math
from ObjectUtils import Zone, Label, Step
from Config import Frame, ParseSetting, SoundSwitch
import copy
from pathlib import Path
import os

def write_video(outVideo, q):
    for frame in q:
        outVideo.write(frame)

def save_video(startTime, q, recordPath): 
    dateText = startTime.strftime('%Y/%m/%d')
    timeText = startTime.strftime('%H:%M:%S')
    dateInfo = dateText.replace('/','')
    timeInfo = timeText.replace(':','')
    
    savePath = os.path.join(recordPath, dateInfo, 'Motion')
    Path(savePath).mkdir(parents=True, exist_ok=True) 
    saveFile = os.path.join(savePath, f'{timeInfo}_MotionFail.avi')

    width =  1280
    height = 720
    fps = 15
    fourcc = cv2.VideoWriter_fourcc('M','P','4','2')
    outVideo = cv2.VideoWriter(saveFile, fourcc, fps, (width, height))

    subProcess = Thread(write_video(outVideo, q))
    subProcess.start()

def save_frame(frame, dateText, recordPath):
    dateInfo = dateText.split(' ')[0].replace('/','')
    timeInfo = dateText.split(' ')[1].replace(':','')
    savePath = os.path.join(recordPath, dateInfo, 'HandsTouch')
    Path(savePath).mkdir(parents=True, exist_ok=True) 
    saveFile = os.path.join(savePath, f'{timeInfo}_HandsFail.jpg')
    cv2.imwrite(saveFile, frame)

def play_alarmSound(soundType):
    if SoundSwitch.flag[soundType]:
        parseCfg = ParseSetting()
        cfg = parseCfg.read('Sound')
        sound = cfg[soundType]
        pygame.mixer.init()
        pygame.mixer.music.load(sound)
        pygame.mixer.music.play()

class PlacementDetection:

    def put_to_zone(self, region, yoloResult):

        if len(yoloResult["hand_PCB_coordinate"]) != 0:
            for yoloRectangle in yoloResult["hand_PCB_coordinate"]:
                for info in Frame.BBoxInfo:
                    if info.zoneName == region:
                        return info.contains_point(yoloRectangle.centerPoint)          
        return False


class ProcessDetection:

    def __init__(self):
        self.jigLabel = Label.initialize()
        self.currentStep = Step.initialize()

        self.__record = {"OK":False, "Alarm":False}
        self.__FrameCount = {"OK":0, "Alarm":0}
        self.__AlarmType = None
        self.__arrivalStep = 0 

        # self.waitStep1 = True 

    def is_first_step(self):
        if self.jigLabel.previous == Label.type1: 
            return True
        return False   
   
    #### ????????? ####
    def is_final_step(self):
        if self.currentStep.number == len(Label.order)-1:
            return True 
        else:
            return False 
    ################

    def __yoloResult_is_none(self, yoloResult):
        if len(yoloResult) == 0:
            return True

    def run(self, yoloResult):
        
        if self.is_final_step == True:
            self.currentStep = Step.initialize()
            
        if self.__yoloResult_is_none(yoloResult["label"]):
            return "NONE", self.currentStep.number
        
        self.jigLabel.update_info(yoloResult, Frame.BBoxInfo)

        if Step.goto_next(self.jigLabel):

            self.currentStep.update_info()

            self.__record = {"OK":False, "Alarm":False}
            self.__FrameCount = {"OK":0, "Alarm":0}

            if self.jigLabel.is_same_label(self.currentStep):
                self.__record["OK"] = True
            else:
                self.currentStep.back_to_previousStep() # ?????????????????????
                if self.jigLabel.is_same_label(self.currentStep):
                    self.__record["OK"] = True
                else:
                    self.__record["Alarm"] = True
                    self.__AlarmType = self.jigLabel.current 

        else:
            if self.__record["OK"]:
                if self.__FrameCount["OK"] > 1: #5
                    if self.__arrivalStep != self.currentStep.number:
                        self.__record["OK"] = False
                        self.__arrivalStep = self.currentStep.number
                        return "NO", self.currentStep.number 
                else:
                    if self.jigLabel.current == self.currentStep.currentLabel:
                        self.__FrameCount["OK"] += 1
            elif self.__record["Alarm"]:
                if self.__FrameCount["Alarm"] > 5: #30
                    self.__record["Alarm"] = False
                    return "YES", self.currentStep.number
                else:
                    if self.jigLabel.current == self.__AlarmType:
                        self.__FrameCount["Error"] += 1 

        return "NONE", self.currentStep.number

class Motion:

    def __init__(self, model, recordPath):
        self.model = model
        self.processDetection = ProcessDetection()
        self.placementDetection = PlacementDetection()
        self.storgeOfVideoFrames = []
        self.haveAlarm = False
        self.needRecord = False
        self.recordPath = recordPath
        self.needJudgementStep4 = False
        self.initialize_parameter()

        self.waitStep1 = True

    def initialize_parameter(self):
        self.processDetection.currentStep = Step.initialize()
        self.centerPointAToD = -10000
        self.processEnd = False
        self.placementOK = {"A":False, "C":False, "D":False}
        self.waitCorrection = {"A":False, "D":False}
        
    def run(self, image):

        self.storgeOfVideoFrames.append(image)

        # ???YOLO
        yoloResult = self.model.detect_motion(image)

        if self.processEnd: # processDetection.is_final_step
            if self.needRecord:
                self.needRecord = False
                temp = copy.deepcopy(self.storgeOfVideoFrames)
                save_video(datetime.now(), temp, self.recordPath) 
            self.initialize_parameter()  
            self.storgeOfVideoFrames.clear()
            self.haveAlarm = False
            self.needRecord = False

        # ?????????????????????
        if self.waitStep1:
            if self.processDetection.currentStep.number == 0 and yoloResult["label"][0] != "no_PCB":
                self.waitStep1 = True
                return ['NONE'], [-1]
            elif self.processDetection.currentStep.number == 0 and yoloResult["label"][0] == "no_PCB":
                self.waitStep1 = False
                self.processDetection.currentStep = Step.initialize()
                self.processDetection.jigLabel = Label.initialize()
                self.storgeOfVideoFrames.clear()
                self.haveAlarm = False
                self.needRecord = False
                self.needJudgementStep4 = False

        isAnomalyStep, numOfCurrentStep = self.processDetection.run(yoloResult) 
        # print(f'isAnomalyStep: {isAnomalyStep}, numOfCurrentStep: {numOfCurrentStep}')

        #################################################### ???"??????"????????????????????? ################################################################    
        ### ???0?????????1??? ###
        if numOfCurrentStep == 0:    # ???1?????????4??????????????????????????????C???
            if not self.placementOK["C"]:
                self.placementOK["C"] = self.placementDetection.put_to_zone(Zone.C, yoloResult)
            else:
                # ??????4???????????????
                if self.needJudgementStep4:
                    self.processEnd = True
                    self.needJudgementStep4 = False
                    return ["NO"], [4]  
        
        ### ?????????????????? : step. 2 ?????????????????? D???A???????????? ###
        elif numOfCurrentStep == 2:

            # ??????????????????Alarm????????????
            if self.haveAlarm:
                if isAnomalyStep == 'NONE':
                    return ["NONE"], [numOfCurrentStep]
                else:
                    return ["YES"], [numOfCurrentStep]

            # ??????A???->D??? (???????????????)
            elif self.waitCorrection['D']:
                if not self.placementOK['D']:
                    if not self.placementOK['A']:
                        self.placementOK['A'] = self.placementDetection.put_to_zone(Zone.A, yoloResult)
                    else:
                        self.centerPointAToD = 0
                        self.placementOK['D'] = self.placementDetection.put_to_zone(Zone.D, yoloResult)
                else:
                    if self.centerPointAToD > 0:
                        self.waitCorrection['D'] = False
                        self.placementOK['A'] = False
                        self.haveAlarm = False
                        return ["NO"], [numOfCurrentStep]
                # ??????A???D???"??????"?????? hand_PCB ??????
                centerPointInA = self.placementDetection.put_to_zone(Zone.A, yoloResult)
                centerPointInD = self.placementDetection.put_to_zone(Zone.D, yoloResult)
                if len(yoloResult["hand_PCB_coordinate"]) > 0 and not centerPointInA and not centerPointInD:
                    self.centerPointAToD += 1

            # ??????D???????????????
            elif not self.placementOK["A"]:   
                if not self.placementOK["D"]: # D????????????
                    self.placementOK["D"] = self.placementDetection.put_to_zone(Zone.D, yoloResult)
                    if self.placementOK["D"]: # D?????????
                        return ["NO"], [numOfCurrentStep]
                    else: # ???????????????D???????????????A????????????????????????
                        if self.placementDetection.put_to_zone(Zone.A, yoloResult):
                            isAnomalyStep = "YES"
                            self.waitCorrection['D'] = True        
                else: # D?????????
                    self.placementOK["A"] = self.placementDetection.put_to_zone(Zone.A, yoloResult)


        ### ??????????????????????????????B???->D??? (???????????????) ###        
        elif numOfCurrentStep == 4:

            # ??????"?????????A?????????D" ??? "?????????D?????????A"
            if self.waitCorrection['D'] or self.waitCorrection['A']:
                if self.processDetection.jigLabel.current == Label.type3: # Label.type3 == "frame"
                    self.processDetection.currentStep.number = 2
                    self.haveAlarm = False
                    return ["NONE"], [-1]
                else:
                    self.processEnd = True
                    return ["YES"], [numOfCurrentStep]
            
            # ??????????????????Alarm????????????
            if self.haveAlarm:
                self.processEnd = True
                return ["YES"], [numOfCurrentStep]

            self.needJudgementStep4 = True
            self.initialize_parameter()  
            return ["NONE"], [-1]


        ######################################################### ???"??????"????????????????????? #############################################################
        # ??????????????????
        if isAnomalyStep == "NO":

            if numOfCurrentStep == 1:
                if not self.placementOK["C"]:
                    if self.needJudgementStep4:
                        self.needJudgementStep4 = False
                        temp = copy.deepcopy(self.storgeOfVideoFrames)
                        save_video(datetime.now(), temp, self.recordPath) 
                        self.storgeOfVideoFrames.clear()
                        thread = Thread(target=play_alarmSound("motion"))
                        thread.start()
                        self.haveAlarm = True
                        self.needRecord = True
                        return ["YES", "YES"], [4, numOfCurrentStep]
                    isAnomalyStep = "YES"
                else:
                    if self.needJudgementStep4:
                        return ["NO", "NO"], [4, numOfCurrentStep]
                    return ["NO"], [numOfCurrentStep]
            
            elif numOfCurrentStep == 3: # ?????????step 3.??????????????????????????????D??????A??????????????? step 2.B->D ???
                
                if self.haveAlarm:
                    if isAnomalyStep == 'NONE':
                        return ["NONE"], [numOfCurrentStep]
                    else:
                        return ["YES"], [numOfCurrentStep]
                
                if not self.placementOK["D"]: # ?????? ??????A?????????D?????????????????? or A??????D??????????????????
                    if self.waitCorrection['D']: # ????????????A?????????D??????????????????
                        return ["YES"], [numOfCurrentStep]
                    thread = Thread(target=play_alarmSound("motion"))
                    thread.start()
                    self.haveAlarm = True
                    self.needRecord = True
                    self.waitCorrection['D'] = True
                    return ["YES", "YES"], [numOfCurrentStep-1, numOfCurrentStep]
                else: # ??????D???????????????
                    if self.placementOK["A"]:
                        self.waitCorrection['A'] = False
                        return ["NO"], [numOfCurrentStep]
                    else:
                        self.waitCorrection['A'] = True
                        isAnomalyStep = "YES"
                        
                        
        # ??????????????????
        if isAnomalyStep == "YES":
            thread = Thread(target=play_alarmSound("motion"))
            thread.start()
            self.haveAlarm = True
            self.needRecord = True
            return [isAnomalyStep], [numOfCurrentStep]

        if isAnomalyStep == 'NONE':
            return [isAnomalyStep], [-1]


class Hands:

    def __init__(self, model, recordPath, pixelThres, degree):
        """initialize parameter

        Args:
            model (torch): YOLOv5 Model
        """        
        self.model = model
        self.pixelThres = pixelThres
        self.clipThres = 15
        self.degree = degree
        self.NGFlag = None 
        self.lastNGImg = None 
        self.lastTimeText = 'init'
        self.preMaskBoxes = []
        self.recordPath = recordPath
        self.frameLock = True
        
    def run(self, image):
        """Stream Hands Abnomal Detection

        Args:
            image (img): perspective image

        Returns:
            img: resultsImg
        """        
        streamImg = image.copy()
        resultsImg = None
        IOUThresh = 0.25
        yoloResult = self.model.detect_hands(image)
        ### initial last NG image
        if self.NGFlag == None:
            self.lastNGImg = cv2.imread('config/idle.jpg')   
        ### last  NG Img resize
        if self.lastNGImg.shape != streamImg.shape:
            self.lastNGImg = cv2.resize(self.lastNGImg, (streamImg.shape[1], streamImg.shape[0]), interpolation=cv2.INTER_AREA)

        ### reload pixel config

        ### draw system time on image
        dateFormat = "%Y/%m/%d %H:%M:%S"
        dateText = datetime.now().strftime(dateFormat)
        cv2.putText(streamImg, dateText, (10, streamImg.shape[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        ### YOLO Predict PCB and Gloves
        if len(yoloResult['PCB'])>0 and len(yoloResult['Gloves'])>1:
            streamImg, inPlateMask, self.NGFlag, centerFlag = self.check_hands_touch(streamImg, yoloResult['PCB'], yoloResult['Gloves'], self.pixelThres, self.clipThres, self.degree)
        else:
            self.NGFlag = False
            centerFlag = False

        ### Record NG Image
        judgeFlag = False
        if self.NGFlag and centerFlag:
            streamImg = self.draw_mask_on_image(streamImg, inPlateMask)
            ### on the same second
            if dateText == self.lastTimeText:   
                if not self.frameLock:
                    ### calcullate IOU between preFrame and currentFrame
                    maskBoxes = self.get_mask_box(inPlateMask)
                    for box1 in maskBoxes:
                        for box2 in self.preMaskBoxes:
                            boxIOU = self.get_iou(box1, box2)
                            if boxIOU >= IOUThresh:    # continuous touching
                                judgeFlag = True
                                self.frameLock = True
                                plot_fail_touch(streamImg, yoloResult['PCB'], self.pixelThres, color=None, line_thickness=3)
                                self.lastNGImg = streamImg
                                save_frame(self.lastNGImg, dateText, self.recordPath)
                                ### Trigger Alert
                                thread = Thread(target=play_alarmSound("hands"))
                                thread.start()
                                break
                        else:
                            continue
                        break
                    self.preMaskBoxes = maskBoxes
            else:
                self.lastTimeText = dateText
                self.preMaskBoxes = []
                self.frameLock = False

        resultsImg = np.hstack((streamImg, self.lastNGImg))   # Concatenate Stream & History Abnomal Image
        return resultsImg, judgeFlag

    ### CV Judge Hands Touch Mask
    def check_hands_touch(self, image, PCBBoxes, GlovesBoxes, pixelThres, clipThres, glovesDegree):
        """hands touch check function

        Args:
            image (img): Original BGR Image 
            PCBBoxes (list): PCB YOLO Bounding Box
            GlovesBoxes (list): Gloves YOLO Bounding Box
            pixelThres (int): Inner Pixels Values
            glovesDegree (float): detection range of gloves

        Returns:
            [img]: image
            [img]: inPlateMask
            [bool]: touchFlag
            [bool]: centerFlag
        """
        touchFlag = False
        centerFlag = False
        (imgH, imgW, _) = image.shape
        ### Mask of PCB???Gloves
        inPlateMask = np.zeros((imgH, imgW), np.uint8)
        PCBMask = np.zeros((imgH, imgW), np.uint8)
        glovesImg = np.zeros((imgH, imgW, 3), np.uint8)
        PCBMask[int(PCBBoxes[1]) + pixelThres + clipThres:int(PCBBoxes[3]) - pixelThres - clipThres, int(PCBBoxes[0]) + pixelThres:int(PCBBoxes[2])- pixelThres] = 255
        ### Mask of GlovesL???GlovesR
        lowerGloves = np.array([150, 150, 150])   # BGR
        upperGloves = np.array([255, 255, 255])   # BGR
        ### Paste to Gloves Mask
        for gloveBBox in GlovesBoxes:
            halfBBoxH = int((1-glovesDegree) * (gloveBBox[3] - gloveBBox[1]))
            # halfbBoxW = int((1-glovesDegree) * (gloveBBox[2] - gloveBBox[0]))
            glovesImg[int(gloveBBox[1]):int(gloveBBox[3] - halfBBoxH), int(gloveBBox[0]):int(gloveBBox[2])] = image[int(gloveBBox[1]):int(gloveBBox[3] - halfBBoxH), int(gloveBBox[0]):int(gloveBBox[2])]
        glovesMask = cv2.inRange(glovesImg, lowerGloves, upperGloves)
        kernel = np.ones((3, 3),np.uint8)
        glovesMask = cv2.morphologyEx(glovesMask, cv2.MORPH_OPEN, kernel)   # Open Operation

        ### Bitwise PCB&Gloves Mask
        inPlateMask = cv2.bitwise_and(PCBMask, glovesMask)
        inPlatePixels = np.sum(inPlateMask)

        ### Place the PCB Center Check
        # radius = 0.5 * pixelThres
        radius = 0.5 * 13
        midPCB = [0.5 * int(PCBBoxes[0] + PCBBoxes[2]), 0.5 * int(PCBBoxes[1] + PCBBoxes[3])]
        midImg = [0.5 * imgW - 1, 0.5 * imgH - 1]
        midDiff = np.array(midPCB) - np.array(midImg)
        midDistance = math.hypot(midDiff[0], midDiff[1])
        centerFlag = True if (midDistance <= radius) else False
        touchFlag = True if inPlatePixels else False

        ### dilate inPlateMask
        kernel = np.ones((3, 3),np.uint8)
        inPlateMask = cv2.dilate(inPlateMask, kernel, iterations=1)

        return image, inPlateMask, touchFlag, centerFlag

    ### Draw Gloves Mask on Image
    def draw_mask_on_image(self, img, maskImg):
        (imgH, imgW, _) = img.shape
        zeroMask = np.zeros((imgH, imgW), np.uint8)
        backgroundMask = cv2.bitwise_not(maskImg)
        backgroundImg = cv2.bitwise_and(img, img, mask=backgroundMask)
        touchImg = cv2.merge([zeroMask, zeroMask, maskImg])
        resultImg = cv2.add(backgroundImg, touchImg)
        return resultImg

    ### get touch image
    def get_mask_box(self, maskImg):
        maskBoxes = []
        # cv2.findContours: ??????ROI?????????
        contours, hierarchy = cv2.findContours(maskImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0, len(contours)):
            # ?????????(x, y)????????????????????????w ?????????(x?????????)???h ?????????(y?????????)
            x, y, w, h = cv2.boundingRect(contours[i])
            maskBoxes.append([x, y, x + w, y + h])
        return maskBoxes

    def get_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou