import io
import configparser
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import cv2
from PIL import Image

from AIModule.ObjectUtils import BBox

class SoundSwitch:
	flag = {"motion":True, "hands":True}

class ConfigType:
	action = "motion_config"
	hand = "hands_config"
	
class ServiceConfig:

	@staticmethod
	def write_config(data, type, recordPath):
		dateFormat = "%Y/%m/%d"
		dateText = datetime.now().strftime(dateFormat)
		### 處理hand、motion txt
		if isinstance(data, list):
			with open('config/{}.txt'.format(type), 'w') as f:
				f.write(('\n').join(data)) 
		else:
			### 處理console txt
			dateInfo = dateText.replace('/','')
			savePath = os.path.join(recordPath, dateInfo) 
			consoleFile = os.path.join(savePath, f'{type}.txt')
			with open(consoleFile, 'a+') as f:
				f.write(data + '\n') 

	#動作流程座標框
	@staticmethod
	def get_bbox_config():
		result = []
		with open('config/{}.txt'.format(ConfigType.action),'r') as f:
			for line in f.readlines():
				content = line.split(" ")
				result.append(BBox.get_from_list(content[0], int(content[1]), int(content[2]), int(content[3]), int(content[4]), 
													int(content[5]), int(content[6]), int(content[7]), int(content[8])))
			return result

    #透視變換四個點
	@staticmethod
	def get_4_coordinate():
		result = []
		with open('config/{}.txt'.format(ConfigType.hand), 'r') as f:
			for line in f.readlines():
				temp = line.split(" ")
				result.append([int(temp[0]), int(temp[1])])
			return result

	#獲得console內容
	@staticmethod
	def get_console_config(type, recordPath):
		date = datetime.now().strftime('%Y%m%d')	
		path = f"{recordPath}/{date}"
		file = os.path.join(path, f'{type}.txt') 
		Path(f"{recordPath}/{date}").mkdir(parents=True, exist_ok=True) #建立資料夾
		Path(file).touch() #建立txt
		with open(file, 'r') as f:
			contents = f.readlines()      
			return contents


class Frame:

	BBoxInfo = ServiceConfig.get_bbox_config() 
	roiPts = ServiceConfig.get_4_coordinate() 

	@staticmethod
	def initialize_label():
		Frame.BBoxInfo = ServiceConfig.get_bbox_config() 
		Frame.roiPts = ServiceConfig.get_4_coordinate() 

	@staticmethod
	def draw_rectangle_in_zone(frame):
		for BBox in Frame.BBoxInfo:
			pts = np.array([[BBox.pointOfTopLeft.x, BBox.pointOfTopLeft.y], [BBox.pointOfTopRight.x, BBox.pointOfTopRight.y],
			 					[BBox.pointOfBottomLeft.x, BBox.pointOfBottomLeft.y], [BBox.pointOfBottomRight.x, BBox.pointOfBottomRight.y]], np.int32)
			cv2.polylines(frame, [pts], True, (0,0,255), 5)
			cv2.putText(frame, BBox.zoneName, (BBox.pointOfTopLeft.x, BBox.pointOfTopLeft.y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)

	@staticmethod
	def transform_virtual_file(frame):
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = Image.fromarray(frame.astype('uint8'))
		file_object = io.BytesIO()
		frame.save(file_object, 'PNG')
		file_object.seek(0)
		return file_object

	@staticmethod
	def encode(frame):
		frame = cv2.imencode('.png', frame)[1].tobytes()
		frame = (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
		return frame

class ParseSetting():

	def __init__(self):
		self.SrvCfg = configparser.ConfigParser()
		self.savePath = './config/setting.cfg'

	def read(self, section, savePath='./config/setting.cfg'):
		self.SrvCfg.read(savePath)
		return self.SrvCfg[section]

