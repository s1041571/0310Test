import cv2
import threading
import datetime, time
from pathlib import Path
import numpy as np
from Config import Frame

class Camera:
    def __init__(self,fps=30, video_source=1):
        # logger.info(f"Initializing camera class with {fps} fps and video_source={video_source}")
        self.fps = fps
        self.isrunning = True
        self.status = False
        self.cap = cv2.VideoCapture(video_source)
        self.source = video_source
        # self.thread = threading.Thread(target=self.update, args=([cap]), daemon=True)
        # self.thread.start()
        # logger.debug("Starting thread")
        # logger.info("Thread started")

    def run(self):
        self.thread = threading.Thread(target=self.update, args=([self.cap]), daemon=True)
        self.thread.start()

    def update(self, cap):
        n = 0
        while cap.isOpened() and self.isrunning:
            n += 1
            cap.grab()
            if n == 1:  # read every 4th frame
                (self.status, self.frame) = cap.retrieve()
                n = 0
            ### RTSP don't need to sleep
            if self.source[:4] != 'rtsp':
                time.sleep(0.05)

    def stop(self):
        # logger.debug("Stopping thread")
        self.isrunning = False

    def get_frame(self):
        ### Camera有讀到影像
        if self.status:
            return cv2.resize(self.frame, (1280, 720))
        ### Camera讀不到影像，讀預設not_found影像
        else:
            with open("config/not_found.jpeg","rb") as f:
                img = f.read()
            return img

def perspective_transform(image, points):
    """perspective transform ROI regison

    Args:
        image (img): origin image
        points (list): (tlp1, trp2, brp3, blp4)

    Returns:
        [img]: warpedImg
    """    
    
    maxHeight, maxWidth = get_perspective_size(points)
    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")
    points = np.array(points, dtype="float32")
    M = cv2.getPerspectiveTransform(points, dst)
    warpedImg = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warpedImg

def get_perspective_size(points):

    (tl, tr, br, bl) = points
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    return maxHeight, maxWidth
