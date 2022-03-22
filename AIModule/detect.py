from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import sys
sys.path.append('./AIModule')

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device


class AbnormalModel:

    def __init__(self):
        self.imgsz = 640
        self.device = None
        ### Model
        self.motionModel = None
        self.handsModel = None
        ### confidence score threshold
        self.hands_conf_thres = 0.80
        self.motion_conf_thres = 0.25
        ### IOU threshold
        self.hands_iou_thres = 0.45
        self.motion_iou_thres = 0.45
        cudnn.benchmark = True  # set True to speed up constant image size inference
    
    
    def load_device(self, device=''):
        """select computing device

        Args:
            device (str, optional): cuda device, i.e. 0 or 0,1,2,3 or cpu. Defaults to ''.
        """        
        if self.device is None:
            self.device = select_device(device)

    def load_motion_model(self, weights, device='', imgsz=640):
        """load motion model (YOLOv5)

        Args:
            weights (.pt): YOLOv5 model weights.
            device (str): cuda device, i.e. 0 or 0,1,2,3 or cpu. Defaults to ''.
            imgsz (int): input resize size. Defaults to 640.
        """        
        self.load_device(device)
        if self.motionModel is None:
            self.motionModel = attempt_load(weights, map_location=self.device)
            self.motionImgsz = check_img_size(imgsz, s=self.motionModel.stride.max())
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
            if self.half:
                self.motionModel.half()  # to FP16

    def load_hands_model(self, weights, device='', imgsz=640):
        """load hands model (YOLOv5)

        Args:
            weights (.pt): YOLOv5 model weights.
            device (str): cuda device, i.e. 0 or 0,1,2,3 or cpu. Defaults to ''.
            imgsz (int): input resize size. Defaults to 640.
        """  
        self.load_device(device)
        if self.handsModel is None:
            self.handsModel = attempt_load(weights, map_location=self.device)
            self.handsImgsz = check_img_size(imgsz, s=self.handsModel.stride.max())
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
            if self.half:
                self.handsModel.half()  # to FP16

    def detect_hands(self, image):
        """hands-touch detection

        Args:
            image (img): perspective camera image

        Returns:
            dict: yoloResult
        """        
        im0s = image 
        ### Get names and colors
        names = self.handsModel.module.names if hasattr(self.handsModel, 'module') else self.handsModel.names
        yoloResult = {"PCB":[], "Gloves":[]}
        ### Run inference
        img = torch.zeros((1, 3, self.handsImgsz, self.handsImgsz), device=self.device)  # init img
        _ = self.handsModel(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        
        ### Padded resize
        img = letterbox(im0s, new_shape=self.handsImgsz)[0]

        ### Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        ### Inference
        # t1 = time_synchronized()
        pred = self.handsModel(img, augment=False)[0]

        ### Apply NMS
        pred = non_max_suppression(pred, self.hands_conf_thres, self.hands_iou_thres, classes=None, agnostic=False)
        # t2 = time_synchronized()

        ### Process detections
        det = pred[0]  # detections per image
        if len(det):
            ### Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            confBase = 0
            ### Write results
            for *xyxy, conf, cls in reversed(det):
                '''
                # label = f'{names[int(cls)]} {conf:.2f}'
                # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                '''
                ### PCB Class bbox
                if cls == 0 and conf > confBase:   # 0:PCB object
                    confBase = conf
                    yoloResult['PCB'] = xyxy
                elif cls == 1:   # 1:Gloves obhect
                    yoloResult['Gloves'].append(xyxy)

            ### Print time (inference + NMS)
            # print(f'Inference Done. ({t2 - t1:.3f}s)')
        return yoloResult

    def detect_motion(self, image):
        """motion detection

        Args:
            image (img): camera view image

        Returns:
            dict: yoloResult
        """        

        im0s = image
        # Get names and colors
        names = self.motionModel.module.names if hasattr(self.motionModel, 'module') else self.motionModel.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        img = torch.zeros((1, 3, self.motionImgsz, self.motionImgsz), device=self.device)  # init img
        _ = self.motionModel(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

        # Padded resize
        img = letterbox(im0s, new_shape=self.motionImgsz)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        yoloResult = {"label":[], "coordinate":[], "hand_PCB_coordinate":[]}

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # t1 = time_synchronized()
        pred = self.motionModel(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.motion_conf_thres, self.motion_iou_thres, classes=None, agnostic=False)
        # t2 = time_synchronized()

        det = pred[0]
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                labelName = f'{names[int(cls)]}'
                # print(labelName)
                labelCoordinate = plot_one_box(xyxy, im0s, label=labelName, color=colors[int(cls)], line_thickness=3)
                if labelName == "hand_PCB":
                    yoloResult["hand_PCB_coordinate"].append(labelCoordinate)
                else:
                    yoloResult["label"].append(labelName)
                    yoloResult["coordinate"].append(labelCoordinate)
            ### Print time (inference + NMS)
            # print(f'Inference Done. ({t2 - t1:.3f}s)')
        return yoloResult
