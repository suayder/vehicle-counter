import numpy as np
import cv2
from utils import get_bbox
from tracker import EuclideanDistTracker
from counter import Counter

path = 'motociata.webm'

rescale_ratio = 0.6

cap = cv2.VideoCapture(path)
#cap.set(cv2.CAP_PROP_POS_MSEC,60000) # start from some milliseconds
ret, first_frame = cap.read()

#variables used in the filter
min_area = 8/186624
max_area = 70/186624

first_frame = cv2.resize(first_frame, None, fx=rescale_ratio, fy=rescale_ratio)
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# cv2.namedWindow('processed')
# cv2.moveWindow('processed', 50,80)
#cv2.namedWindow('original')
#cv2.moveWindow('original', 500,80)

h,w = first_frame.shape
line = np.array([(w*0.335,h*0.6),(w*0.656,h*0.6)], dtype=int)
line_2 = np.array([(w*0.315,h*0.65),(w*0.656,h*0.65)], dtype=int)
polly = np.array([(w*0.476,h*0.339), (w*0.20,h*0.833), (w*0.650,h*0.833), (w*0.64,h*0.385)], dtype=int)

mask = np.zeros_like(first_frame, dtype=np.uint8)
cv2.fillPoly(mask, [polly],color=(255,255,255))

structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,5))

backSub = cv2.createBackgroundSubtractorMOG2(history=150, detectShadows=True)
backSub.setNMixtures(3)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

dist_tracker = EuclideanDistTracker()
counter = Counter(14, 2, 20)
frame_counter = 1
while 1:
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter+=1
    frame = cv2.resize(frame, None, fx=rescale_ratio, fy=rescale_ratio)
    color_frame = frame.copy()
    frame = cv2.GaussianBlur(frame,(5,5),5)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Detect object
    bgmask = backSub.apply(frame, learningRate=0.001)
    _, bgmask = cv2.threshold(bgmask, 128,255, cv2.THRESH_BINARY)

    bgmask = cv2.bitwise_and(bgmask, mask)
    bgmask = cv2.medianBlur(bgmask,5)
    #bgmask = cv2.morphologyEx(bgmask, cv2.MORPH_ERODE, structuring_element,iterations=1)
    bgmask = cv2.morphologyEx(bgmask, cv2.MORPH_OPEN, structuring_element,iterations=1)
    rects = get_bbox(bgmask, min_area, max_area)
    
    #tracker object
    boxes_ids = dist_tracker.update(rects)
    center_points = dist_tracker.center_points

    #object count
    region_points = {}
    for i in center_points:
        if center_points[i][1]>line[0][1] and center_points[i][1]<line_2[0][1]:
            region_points[i] = center_points[i]
    counter.update(region_points, frame_counter)
    


    #Drawing functions
    bgmask = cv2.cvtColor(bgmask, cv2.COLOR_GRAY2BGR)

    color_frame = cv2.putText(color_frame,
                                f'MOTOS: {str(counter.count)}',
                                (w//2,50),
                                cv2.FONT_HERSHEY_PLAIN,
                                3,
                                (255,0,0),
                                3,
                                cv2.LINE_AA)

    for rect in boxes_ids:
        cv2.rectangle(bgmask, (rect[0],rect[1]), (rect[2]+rect[0],rect[3]+rect[1]),(0,255,0), 1)
        cv2.rectangle(color_frame, (rect[0],rect[1]), (rect[2]+rect[0],rect[3]+rect[1]),(0,255,0), 1)
        color_frame = cv2.putText(color_frame,
                                  str(rect[4]),
                                  (rect[0],rect[1]-20),
                                  cv2.FONT_HERSHEY_PLAIN,
                                  1,
                                  (255,0,0),
                                  2,
                                  cv2.LINE_AA)

    cv2.polylines(color_frame, [polly], True, (0,0,255), 3)
    cv2.line(color_frame,line[0], line[1], (255,0,0), 2)
    cv2.line(color_frame,line_2[0], line_2[1], (0,255,0), 2)

    # cv2.imshow('processed', bgmask)
    # cv2.imshow('original', color_frame)
    # key = cv2.waitKey(50) & 0xff
    # if key==27:
    #    break

    #save frame
    if frame_counter>total_frames-2 or frame_counter%1000==0:
        cv2.imwrite('motociata_count.png',color_frame)
        print('ready:', ((frame_counter*100)/total_frames), '%')

cap.release()
cv2.destroyAllWindows()