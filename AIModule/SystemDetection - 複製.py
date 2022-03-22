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
from Config import Frame
import copy
from pathlib import Path
import os

from Config import Frame, ParseSetting

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

        self.OKRecord = False
        self.ErrorRecord = False
        self.OKCount = 0
        self.ErrorCount = 0
        self.ErrorType = None
        self.arrivalStep = 0

    def is_first_step(self):
        if self.jigLabel.previous == Label.type1: 
            return True
        return False   
   
    #### 有疑問 ####
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

            if self.jigLabel.is_not_same_label(self.currentStep) == "YES":
                self.currentStep.back_to_previousStep() # 不更新目前狀態
                if self.jigLabel.is_not_same_label(self.currentStep) == "YES":
                    self.ErrorRecord = True
                    self.ErrorType = self.jigLabel.current  
                else:
                    self.ErrorRecord = False
                    self.ErrorCount = 0  
            else:
                self.OKRecord = True
                self.OKCount += 1
        
        else:
            if self.OKRecord:
                if self.OKCount > 5:
                    if not self.arrivalStep == self.currentStep.number:
                        self.OKRecord = False
                        self.OKCount = 0
                        self.arrivalStep = self.currentStep.number
                        return "NO", self.currentStep.number
                else:
                    if self.jigLabel.current == self.currentStep.currentLabel:
                        self.OKCount += 1
            elif self.ErrorRecord:
                if self.ErrorCount > 30:  
                    self.ErrorRecord = False
                    self.ErrorCount = 0
                    return "YES", self.currentStep.number
                else:
                    if self.jigLabel.current == self.ErrorType:
                        self.ErrorCount += 1 
        
        return "NONE", self.currentStep.number

class Motion:

    def __init__(self, model, recordPath):
        self.model = model
        self.processDetection = ProcessDetection()
        self.placementDetection = PlacementDetection()
        self.q = []
        self.temp1 = False
        self.haveAlarm = False
        self.needRecord = False
        self.waitStep1 = True
        self.initialize_parameter()
        self.recordPath = recordPath
        self.getFirstStep = False

    def initialize_parameter(self):
        # self.q.clear()
        self.processDetection.currentStep.number = 0
        self.centerPointAToD = 0
        self.processEnd = False
        self.placementOK = {"A":False, "C1":False, "C2":False, "D":False}
        self.waitCorrection = {"A":False, "D":False}
        self.getFirstStep = False
        
    def run(self, image):

        self.q.append(image)

        # 跑YOLO
        yoloResult = self.model.detect_motion(image)
        # print(f'yoloResult: {yoloResult}', self.processDetection.currentStep.number)

        if self.processDetection.currentStep.number == 0 and self.waitStep1:
            if len(yoloResult["coordinate"]) == 1: 
                for rectangle in Frame.BBoxInfo:
                    if rectangle.zoneName == "Jig":
                        BBox = rectangle

                if not (BBox.contains_point(yoloResult["coordinate"][0].centerPoint) and yoloResult["label"][0] == "no_PCB"):
                    self.q = []
                    self.temp1 = False
                    self.haveAlarm = False
                    self.needRecord = False
                    self.centerPointAToD = 0
                    self.processEnd = False
                    self.placementOK = {"A":False, "C1":False, "C2":False, "D":False}
                    self.waitCorrection = {"A":False, "D":False}
                    return ['NONE'], [-1]
                else:
                    self.waitStep1 = False    
                    self.processDetection.jigLabel.previous = Label.type1
                    self.processDetection.jigLabel.current = Label.type1
            else:
                return ['NONE'], [-1]


        if self.processEnd: # processDetection.is_final_step
            if self.needRecord:
                self.needRecord = False
                temp = copy.deepcopy(self.q)
                save_video(datetime.now(), temp, self.recordPath) 
            self.initialize_parameter()  
            self.q.clear()
            self.haveAlarm = False
            self.needRecord = False

        if self.processDetection.currentStep.number == 0 and yoloResult["label"][0] != "no_PCB" and not self.getFirstStep:
            self.processDetection.currentStep.number = 0
            self.processDetection.currentStep.previousLabel = Label.type1
            self.processDetection.currentStep.currentLabel = Label.type1
            return ['NONE'], [-1]
        elif self.processDetection.currentStep.number == 0 and yoloResult["label"][0] == "no_PCB":
            self.processDetection.jigLabel.previous = Label.type1
            self.processDetection.jigLabel.current = Label.type1
            self.getFirstStep = True

        isAnomalyStep, numOfCurrentStep = self.processDetection.run(yoloResult)    

        #################################################### 用放置去判斷是否正確 ################################################################    
        if self.processDetection.currentStep.number == 0:
            if not self.placementOK["C2"] and self.temp1:
                C2_placementOK = self.placementDetection.put_to_zone(Zone.C, yoloResult)
                if C2_placementOK:
                    self.placementOK["C2"] = True
                    self.processEnd = True
                    self.temp1 = False
                    return ["NO"], [4]
            if not self.placementOK["C1"]:
                C1_placementOK = self.placementDetection.put_to_zone(Zone.C, yoloResult)
                if C1_placementOK:
                    self.placementOK["C1"] = True
                    # return ["NO"], [numOfCurrentStep]

        # 目前步驟待定 : step. 2 要不斷去判斷 D與A區的狀態
        elif self.processDetection.currentStep.number == 2:

            if self.haveAlarm:
                if isAnomalyStep == 'NONE':
                    return ["NONE"], [numOfCurrentStep]
                else:
                    return ["YES"], [numOfCurrentStep]

            # 判斷A區->D區 (第二步補救)
            if self.waitCorrection['D']:
                if not self.placementOK['A']:
                    A_placementOK = self.placementDetection.put_to_zone(Zone.A, yoloResult)
                    if A_placementOK:
                        self.placementOK['A'] = True
                        self.centerPointAToD = 0
                elif self.placementOK['A']:
                    D_placementOK = self.placementDetection.put_to_zone(Zone.D, yoloResult)
                    if D_placementOK and self.centerPointAToD > 0: #已正確修正(A區->D區)
                        self.waitCorrection['D'] = False
                        self.placementOK['A'] = False
                        self.placementOK['D'] = True
                        self.haveAlarm = False
                        return ["NO"], [numOfCurrentStep]

                centerPointInA = self.placementDetection.put_to_zone(Zone.A, yoloResult)
                centerPointInD = self.placementDetection.put_to_zone(Zone.D, yoloResult)
                if len(yoloResult["hand_PCB_coordinate"]) > 0 and not centerPointInA and not centerPointInD:
                    self.centerPointAToD += 1

            elif not self.placementOK["D"]:
                D_placementOK = self.placementDetection.put_to_zone(Zone.D, yoloResult)
                if D_placementOK:
                    self.placementOK["D"] = True
                    return ["NO"], [numOfCurrentStep]
                else: # 若還沒放到D區就先放到A區時，就判流程錯
                    A_placementOK = self.placementDetection.put_to_zone(Zone.A, yoloResult)
                    if A_placementOK:
                        isAnomalyStep = "YES"
                        self.waitCorrection['D'] = True
                        
            elif self.placementOK["D"] and not self.placementOK["A"]:
                A_placementOK = self.placementDetection.put_to_zone(Zone.A, yoloResult)
                if A_placementOK:
                    self.placementOK["A"] = True

        # 判斷有無回到第二步，B區->D區 (第三步補救)        
        elif self.processDetection.currentStep.number == 4:

            if self.waitCorrection['D'] or self.waitCorrection['A']:
                if self.processDetection.jigLabel.current == Label.type3:
                    self.processDetection.currentStep.number = 2
                    self.haveAlarm = False
                    return ["NO"], [self.processDetection.currentStep.number]
                else:
                    isAnomalyStep = "YES"
                    self.processEnd = True
            # else:
            #     if not self.placementOK["C2"]:
            #         C2_placementOK = self.placementDetection.put_to_zone(Zone.C, yoloResult)
            #         if C2_placementOK:
            #             self.placementOK["C2"] = True
            #             self.processEnd = True
            #             return ["NO"], [numOfCurrentStep]
            
            if self.haveAlarm:
                self.processEnd = True
                if isAnomalyStep == 'NONE':
                    return ["NONE"], [numOfCurrentStep]
                else:
                    return ["YES"], [numOfCurrentStep]

            if isAnomalyStep != "YES":
                self.temp1 = True
                self.initialize_parameter()  
                # self.processEnd = True
                return ["NONE"], [-1]

        ######################################################### 用流程去判斷是否正確 #############################################################
        # 目前步驟正確
        if isAnomalyStep == "NO":

            if self.processDetection.is_first_step():
                pass

            if self.processDetection.currentStep.number == 1:
                list1 = []
                list2 = []
                if not self.placementOK["C2"] and self.temp1:
                    self.temp1 = False
                    list1.append("YES")
                    list2.append(4)
                    temp = copy.deepcopy(self.q)
                    save_video(datetime.now(), temp, self.recordPath) 
                    self.q.clear()
                    self.haveAlarm = False
                if not self.placementOK["C1"]:
                    self.temp1 = False
                    isAnomalyStep = "YES"
                    list1.append("YES")
                    list2.append(1)

                    thread = Thread(target=play_alarmSound("motion"))
                    thread.start()
                    self.haveAlarm = True
                    self.needRecord = True

                    return list1, list2
                else:
                    self.temp1 = False
                    list1.append("NO")
                    list2.append(1)
                    # return isAnomalyStep, numOfCurrentStep
                    return list1, list2
                

            if self.processDetection.currentStep.number == 2: # 交由PlacementDetection判斷
                pass
            
            if self.processDetection.currentStep.number == 3: # 流程到step 3.，但過程中面板沒有到D區和A區時，就判 step 2.B->D 錯
                
                if self.haveAlarm:
                    if isAnomalyStep == 'NONE':
                        return ["NONE"], [numOfCurrentStep]
                    else:
                        return ["YES"], [numOfCurrentStep]

                if self.waitCorrection['D']: # 代表有過A區、但D區都沒放面板
                    isAnomalyStep = "YES"
                elif not self.placementOK["D"]: # 代表A區、D區都沒放面板
                    # isAnomalyStep = "YES"
                    # numOfCurrentStep = numOfCurrentStep-1
                    # self.waitCorrection['D'] = True
                    thread = Thread(target=play_alarmSound("motion"))
                    thread.start()
                    self.haveAlarm = True
                    self.needRecord = True
                    self.waitCorrection['D'] = True
                    return ["YES", "YES"], [numOfCurrentStep-1, numOfCurrentStep]
                else: # 代表D區有放面板
                    if self.placementOK["A"]:
                        self.waitCorrection['A'] = False
                        return [isAnomalyStep], [numOfCurrentStep]
                    else:
                        isAnomalyStep = "YES"
                        self.waitCorrection['A'] = True

            elif self.processDetection.currentStep.number == 4: # 交由PlacementDetection判斷
                pass

            else:
                pass

        # 目前步驟錯誤
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
        ### Mask of PCB、Gloves
        inPlateMask = np.zeros((imgH, imgW), np.uint8)
        PCBMask = np.zeros((imgH, imgW), np.uint8)
        glovesImg = np.zeros((imgH, imgW, 3), np.uint8)
        PCBMask[int(PCBBoxes[1]) + pixelThres + clipThres:int(PCBBoxes[3]) - pixelThres - clipThres, int(PCBBoxes[0]) + pixelThres:int(PCBBoxes[2])- pixelThres] = 255
        ### Mask of GlovesL、GlovesR
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
        # cv2.findContours: 找到ROI的輪廓
        contours, hierarchy = cv2.findContours(maskImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0, len(contours)):
            # 輸出：(x, y)矩形左上角座標、w 矩形寬(x軸方向)、h 矩形高(y軸方向)
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