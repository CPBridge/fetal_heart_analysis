#!/usr/bin/python

import numpy as np
import cv2
import sys

RETURN_KEY = 10
coords = []

# Mouse click callback
def on_mouse(event , x, y, flags, params):
	global coords, disp, img

	if event == cv2.EVENT_LBUTTONDOWN:
		if len(coords) >= 4:
			coords = []
			disp = img.copy()
			cv2.imshow("Mask Crop Tool", disp)

		coords.append([x,y])

		if len(coords) == 4:
			cv2.line(disp,tuple(coords[0]),tuple(coords[1]),(0,0,255))
			cv2.line(disp,tuple(coords[1]),tuple(coords[2]),(0,0,255))
			cv2.line(disp,tuple(coords[2]),tuple(coords[3]),(0,0,255))
			cv2.line(disp,tuple(coords[3]),tuple(coords[0]),(0,0,255))
			cv2.imshow("Mask Crop Tool", disp)


# Get the filename and open the video
vidobj = cv2.VideoCapture(sys.argv[1])
ret, img = vidobj.read()
disp = img.copy()

if not ret:
	print "Couldn't open the video"


cv2.namedWindow("Mask Crop Tool")
cv2.setMouseCallback("Mask Crop Tool", on_mouse, 0)

# Display 
key = 0
cv2.imshow("Mask Crop Tool", disp)

while key != RETURN_KEY :
	key = cv2.waitKey(0)

# Get the coordinates into the format fillPoly expects
coordsarray = np.array(coords,np.int32)
coordsarray = coordsarray.reshape(-1,1,2)

# Fill the polygon to create a mask
mask = np.zeros(img.shape,np.uint8)
mask = cv2.fillPoly(mask,[coordsarray],(255,255,255))
cv2.imshow("Mask Crop Tool", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Make sure that the edges of the image are all black -
# This is important for distance transforms
mask[:][0] = 0
mask[:][-1] = 0
mask[0][:] = 0
mask[-1][:] = 0

# Write the result to file
cv2.imwrite(sys.argv[2],mask)

vidobj.release()