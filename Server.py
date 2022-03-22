from logging import exception
import re
from unittest import removeResult
from flask import Flask, render_template, send_from_directory, Response, request, json, send_file, redirect, url_for
from pathlib import Path
from Camera import Camera, perspective_transform
from Logger import *
from datetime import datetime
import cv2
import copy
import win32api

from Config import SoundSwitch, Frame, ServiceConfig, ConfigType, ParseSetting
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from AIModule.SystemDetection import Motion, Hands
from AIModule.detect import AbnormalModel

app = Flask(__name__)
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    pass

def iteratate_motion_frame(camera):
    global MotionFrame
    while True:
        MotionFrame = camera.get_frame()
        # try:
        #     isAnomalyStep, numOfCurrentStep = motionDetection.run(MotionFrame)
        #     change_html_stepState(isAnomalyStep, numOfCurrentStep)
        # except:
        #     continue     
        frame = copy.deepcopy(MotionFrame)
        Frame.draw_rectangle_in_zone(frame) # 畫作業區框框
        frame = Frame.encode(frame)
        yield frame

def iteratate_hand_frame(camera):
    global HandFrame, HandAlarmCount
    
    while True:
        HandFrame = camera.get_frame()
        try:
            frame = perspective_transform(HandFrame, Frame.roiPts) #透視變換函數
            frame, judgeFlag = handsDetection.run(frame)
        except:
            judgeFlag = False
        frame = Frame.encode(frame)
        if judgeFlag:
            HandAlarmCount += 1
        yield frame

def InitGlbVar():
    global MotionFrame, HandFrame, HandAlarmCount, motionDetectionOpened
    HandAlarmCount = 0
    motionDetectionOpened = True
    
### 主頁面進入點
@app.route("/entrypoint", methods=['GET'])
def entrypoint():
    Frame.initialize_label() # 子頁面修改label後，要更新model作業區位置
    motionDetection.processDetection.currentStep.number = 0
    motionDetection.waitStep1 = True
    return render_template("index.html", user_time=system_time_info())

### 跳轉子頁面
@app.route("/sec_page", methods=['GET'])
def second_entrypoint():
    Logger.info("Requested /second_page")
    return render_template("sec_page.html")

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template("login.html")
    user_id = request.form['user_id']
    if (user_id =='admin') and (request.form['password'] == 'zn123456'):
        return redirect(url_for('entrypoint'))
    
    win32api.MessageBox(0, "無法識別該用戶帳號及密碼\nThe username or password is not recognized.", "Authentication Error", 0x1000) #4096
    return render_template('login.html')
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering or Chrome Frame,
    and also to cache the rendered page for 10 minutes
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers["Cache-Control"] = "public, max-age=0"
    return r

### 取得系統時間
@app.route("/now_time.txt")
def system_time_info():
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S")

### 回傳動作流程camera的frame
@app.route("/video_feed_0")
def video_feed_0():
    return Response(iteratate_motion_frame(camera0),
		mimetype="multipart/x-mixed-replace; boundary=frame")

### 第1個camera的串流含式(手部)
@app.route("/video_feed_1")
def video_feed_1():
    return Response(iteratate_hand_frame(camera1),
		mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/sound_switch", methods=['POST']) 
def sound_switch():
    if request.method == "POST":
        soundType = request.get_json(force=True)
        SoundSwitch.flag[soundType] = not SoundSwitch.flag[soundType]

@app.route("/open_motion_detection", methods=['POST']) 
def open_motion_detection():
    global motionDetectionOpened
    if request.method == "POST":
        motionDetectionOpened = request.get_json(force=True)
        if motionDetectionOpened:
            motionDetection.processDetection.currentStep.number = 0
            motionDetection.waitStep1 = True

@app.route("/motion_result", methods=['POST']) 
def motion_result():
    if request.method == "POST":
        try:
            if motionDetectionOpened:
                isAnomalyStep, numOfCurrentStep = motionDetection.run(MotionFrame)
                return json.dumps({'step_result':isAnomalyStep, 'step_num':numOfCurrentStep})
            return json.dumps({'step_result':"NONE", 'step_num':-1}) 
        except:
            return json.dumps({'step_result':"NONE", 'step_num':-1}) 


### 取得動作狀態的函數，給前端使用(前端會一直呼叫這個函數)
@app.route("/hands_result", methods=['POST']) 
def hands_result():
    global HandAlarmCount
    if request.method == "POST":
        if HandAlarmCount > 0:
            HandAlarmCount = HandAlarmCount - 1
            return json.dumps(True)
        else:
            return json.dumps(False) 

### 跳子頁面要傳當前圖片
@app.route("/currentMotionFrame", methods=['GET'])
def get_currentMotionFrame():
	global MotionFrame
	frame = copy.deepcopy(MotionFrame)
	frame = cv2.resize(frame, (int(frame.shape[1]/1.6), int(frame.shape[0]/1.6)), interpolation=cv2.INTER_AREA)
	file_object = Frame.transform_virtual_file(frame)
	return send_file(file_object, mimetype='image/PNG')

@app.route("/currentHandFrame", methods=['GET'])
def get_currentHandFrame():
	global HandFrame
	frame = copy.deepcopy(HandFrame)
	frame = cv2.resize(frame, (int(frame.shape[1]/1.6), int(frame.shape[0]/1.6)), interpolation=cv2.INTER_AREA)
	file_object = Frame.transform_virtual_file(frame)
	return send_file(file_object, mimetype='image/PNG')

### 前端(子頁面)的透視變換
@app.route("/perspective_transformation", methods=['GET', 'POST'])
def perspective_transformation():
    coordinates = ServiceConfig.get_4_coordinate()
    warpedImg = perspective_transform(HandFrame, coordinates)
    file_object = Frame.transform_virtual_file(warpedImg)
    return send_file(file_object, mimetype='image/PNG')

### 作業區矩形座標寫入txt
@app.route("/action_submit", methods=["POST"])
def action_submit():
    if request.method == "POST":
        data = request.get_json(force=True)
        ServiceConfig.write_config(data, ConfigType.action, recordPath)
        return "OK"
    else:
        return "OK"

### 手部座標點寫入txt
@app.route("/hand_submit", methods=["POST"])
def hand_submit():
    if request.method == "POST":
        data = request.get_json(force=True)
        ServiceConfig.write_config(data, ConfigType.hand, recordPath)
        return "OK"
    else:
        return "OK"

### (子頁面)console寫入txt
@app.route("/download_console", methods=["POST"])
def download_console():
    if request.method == "POST":
        data = request.get_json(force=True)
        ServiceConfig.write_config(data['text'], data['consoleType'], recordPath)    
        return "OK"
    else:
        return "OK" 

### (主、子頁面)console讀取console_config.txt
@app.route("/read_console_config", methods=['POST'])
def read_consoleInfo():
    consoleType = request.get_json(force=True)
    data = ServiceConfig.get_console_config(consoleType, recordPath)
    return json.dumps(data) 

### (子頁面)讀取motion_config.txt 資料傳到前端
@app.route("/read_motion_config", methods=['POST'])
def read_motion_config():
    if request.method == "POST":
        datas = ServiceConfig.get_bbox_config()
        temp = []
        for data in datas:
            temp.append(data.__dict__)
        return json.dumps(temp)
    else:
        return "OK"

### (子頁面)讀取 hands_config.txt 資料傳到前端
@app.route("/read_hands_config", methods=['POST'])
def read_hands_config():
    if request.method == "POST":
        data = ServiceConfig.get_4_coordinate()
        return json.dumps(data)
    else:
        return "OK"

if __name__=="__main__":

    ### set parameter
    InitGlbVar()
    parseCfg = ParseSetting()
    source = parseCfg.read('Source')
    target = parseCfg.read('Target')
    weights = parseCfg.read('Weights')
    handsSetting = parseCfg.read('HandsSetting')

    handsSource = source['hands']
    motionSource = source['motion']
    port = int(source['port'])
    host = source['host']
    recordPath = target['recordPath']
    motionWeight = weights['motion']
    handsWeight = weights['hands']
    handsThres = int(handsSetting['pixels'])
    handsDegree = float(handsSetting['degree'])
    
    ### Log Module Config
    Logger.config(
        logTypes=LogType.Console | LogType.File,
        consoleLogConfig=ConsoleLogConfig(
            level=LogLevel.WARNING,
        ),
        fileLogConfig=FileLogConfig(
            level=LogLevel.INFO,
            newline=False,
            dirname=recordPath,
            suffix="debug",
        )
    )
    
    ###
    print("Welcome to AUO Vision Guard V1.2...")
    ### set AI model
    AIModel = AbnormalModel()
    Logger.info("AI Model Build...")
    AIModel.load_motion_model(motionWeight)
    Logger.info(f"Loading Motion AI Model from {motionWeight}...OK")
    AIModel.load_hands_model(handsWeight)
    Logger.info(f"Loading Hands AI Model from {handsWeight}...OK")
    motionDetection = Motion(AIModel, recordPath)
    handsDetection = Hands(AIModel, recordPath, handsThres, handsDegree)

    ### 初始化兩個camera
    camera0 = Camera(video_source=motionSource)
    camera0.run() # thread 1
    Logger.info(f"Loading Camera0 from {motionSource}")
    camera1 = Camera(video_source=handsSource)
    camera1.run() # thread 2
    Logger.info(f"Loading Camera1 from {handsSource}")
    ### Build Web Server
    Logger.info("Starting Web Server...")
    app.run(host=host, port=port)

    
