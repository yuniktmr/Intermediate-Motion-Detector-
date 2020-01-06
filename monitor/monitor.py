import argparse
from imutils.video import VideoStream
import cv2
import numpy as np
import time
import os
from config.conf import Conf
from keyclipwriter import KeyClipWriter
import sys
import datetime
import imutils
import signal

def signal_handler(sig,frame):
	print("CTRL + c was pressed")
	print("INFO your files are saved in `{}` directory".format(conf["output_path"]))
	
	if kcw.recording:
		kcw.finish()
	sys.exit(0)

ap = argparse.ArgumentParser()
ap.add_argument("-c","--config",required=True, help = "path to config file")
ap.add_argument("-v","--video",type = str,help="path to source video")

args = vars(ap.parse_args())

conf = Conf(args["config"])

if args.get("video",None) is None:
	print("INFO starting videoStream")
	vs = VideoStream(usePiCamera = True).start()
	time.sleep(3.0)
else:
	vs = cv2.VideoCapture(args["video"])

OPENCV_BG_SUBTRACTORS = {"CNT": cv2.bgsegm.createBackgroundSubtractorCNT,
			"MOG": cv2.bgsegm.createBackgroundSubtractorMOG,
			"GMG": cv2.bgsegm.createBackgroundSubtractorGMG,
			"GSOC": cv2.bgsegm.createBackgroundSubtractorGSOC,
			"LSBP": cv2.bgsegm.createBackgroundSubtractorLSBP}

fgbg = OPENCV_BG_SUBTRACTORS[conf["bg_sub"]]()

ekernel = np.ones(tuple(conf["erode"]["kernel"]),"uint8")
dkernel = np.ones(tuple(conf["dilate"]["kernel"]),"uint8")

kcw = KeyClipWriter(bufSize = conf["keyclipwriter_buffersize"])
framesWithoutMotion = 0
framesSinceSnap = 0

signal.signal(signal.SIGINT, signal_handler)
images = "..and images.." if conf["write_snaps"] else ".."
#print("[INFO] detecting motion and storing videos {}").format(images)

while True:
	fullFrame = vs.read()
	
	if fullFrame is None:
		break
	fullFrame = fullFrame[1] if args.get("video", False) else fullFrame
	framesSinceSnap += 1
	
	frame = imutils.resize(fullFrame, width = 500)
	mask = fgbg.apply(frame)

	mask = cv2.erode(mask, ekernel, iterations = conf["erode"]["iterations"])
	mask = cv2.dilate(mask, dkernel, iterations = conf["dilate"]["iterations"])
	cv2.imshow("test",mask)
	cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	
	motionThisFrame = False
	
	for c in cnts:
		((x,y),radius) = cv2.minEnclosingCircle(c)	
		(rx, ry, rw, rh) = cv2.boundingRect(c)

		(x,y,radius) = [int(v) for v in (x,y,radius)]
		
		if radius < conf["min_radius"]:
			continue

		timestamp = datetime.datetime.now()
		timestring = timestamp.strftime("%Y%m%d-%H%M%S")
	
		motionThisFrame = True
		framesWithoutMotion = 0
			
		if conf["annotate"]:
			cv2.circle(frame,(x,y),radius,(0,0,255),2)
			cv2.rectangle(frame, (rx,ry),(rx+rw,ry+rh),(0,255,0),2)
		writeFrame = framesSinceSnap >= conf["frames_between_snaps"]
		if conf["write_snaps"] and writeFrame:
			snapPath = os.path.sep.join([conf["output_path"], timestring])
			cv2.imwrite(snapPath + ".jpg", fullFrame)

			framesSinceSnap = 0
		if not kcw.recording:
			videoPath = os.path.sep.join([conf["output_path"], timestring])

			fourcc = cv2.VideoWriter_fourcc(*conf["codec"])
			kcw.start("{}.avi".format(videoPath),fourcc,conf["fps"])
	if not motionThisFrame:
		framesWithoutMotion += 1
	kcw.update(frame)
		
	noMotion = framesWithoutMotion >= conf["keyclipwriter_buffersize"]

	if kcw.recording and noMotion:
		kcw.finish()
	if conf["display"]:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break
if kcw.recording():
	kcw.finish()
vs.stop() if not args.get("video",False) else vs.release()




