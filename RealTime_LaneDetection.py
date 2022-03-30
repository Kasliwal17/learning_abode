import cv2
import numpy as np
import os
import re

import tqdm.notebook
import tqdm
import matplotlib.pyplot as plt
col_frames = os.listdir('/home/aditya/frames')###we get a list of names  of files in the directory provided
col_frames.sort(key=lambda x: int(re.sub('\D', '', x)))##first remove all non numbers from list and then sort it numerically

colour_im = []
for i in tqdm.notebook.tqdm(col_frames):
    image = cv2.imread('/home/aditya/frames/'+i)### +i adds the name of photo file to the general directory so that we can get the complete directory of an image
    colour_im.append(image)

index=135

plt.figure(figsize=(10,10))
plt.imshow(colour_im[index][:,:,0], cmap= "gray")
plt.show()



zeroarray = np.zeros_like(colour_im[index][:,:,0])
polygon = np.array([[50,270], [220,160], [360,160], [480,270]])
cv2.fillConvexPoly(zeroarray, polygon, 1)### this will create a polygon of shape provided

img = cv2.bitwise_and(colour_im[index][:,:,0], colour_im[index][:,:,0], mask=zeroarray)##here we apply mask on our image
plt.figure(figsize=(10,10))
plt.imshow(img, cmap= "gray")
plt.show()

ret, thresh = cv2.threshold(img, 130, 200, cv2.THRESH_BINARY)  ## simple thresholding our image,any pixel with intensity greater than 130 is set to 200

# plot image
plt.figure(figsize=(10,10))
plt.imshow(thresh, cmap= "gray")
plt.show()

lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=200)

# create a copy of the original frame
dmy = colour_im[index][:,:,0].copy()

# draw Hough lines
for line in lines:
  x1, y1, x2, y2 = line[0]
  cv2.line(dmy, (x1, y1), (x2, y2), (255, 255, 255), 3)

# plot frame
plt.figure(figsize=(10,10))
plt.imshow(dmy)
plt.show()






