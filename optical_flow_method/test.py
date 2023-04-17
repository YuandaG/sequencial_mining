import cv2 as cv
import numpy as np
 
# cap = cv.VideoCapture("clip.mp4")
cap = cv.VideoCapture("clip.mp4")
 
ret, first_frame = cap.read()
 
vedio_width = int(cap.get(3))
vedio_hight = int(cap.get(4))
vedio_fps = int(cap.get(5))
 
video_cod = cv.VideoWriter_fourcc(*'MP4V')
video_output= cv.VideoWriter('flow_day.mp4',
                      video_cod,
                      vedio_fps,
                      (vedio_width,vedio_hight))
 
i = 0
 
while(cap.isOpened()):
    _, frame = cap.read()
    cv.imshow("input", frame)
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
 
    # define blue color range
    light_blue = np.array([20,40,70])
    dark_blue = np.array([255,255,255])
 
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, light_blue, dark_blue)
 
    # Bitwise-AND mask and original image
    output = cv.bitwise_and(frame,frame, mask= mask)
      
    cv.imshow('output',output)
    video_output.write(output)
    # Press Q on keyboard to stop recording
    if cv.waitKey(1) & 0xFF == ord('Q'):
      break
     
    # i += 1
 
    # if i == 500:
    #   break
 
 
cap.release()
video_output.release()
cv.destroyAllWindows()