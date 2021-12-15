import numpy as np
import cv2

def get_bbox(image, min_area, max_area):
    """
    find contourns and return the bounding boxes

    Args:
        image: is a binary image with objects
    """
    contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rects = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        #print(area)
        if area<(image.size*min_area) or area>(image.size*max_area):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        #rects.append([int(x), int(y), int(x+w), int(y+h)])
        rects.append([int(x), int(y), int(w), int(h)])
    #ar = np.array(rects)
    #m = (np.abs(ar[:,0]-ar[:,2])*np.abs(ar[:,1]-ar[:,3])).mean()
    #print(m)
    return np.array(rects)