# Copyright 2019 by Woosung Jang, The Yonsei Univ. ME. KBD Lab.

import cv2 as cv2
import numpy as np
from scipy import ndimage
from PIL import Image,ImageFont,ImageDraw

def initialize():
	img = cv2.imread('~.jpg')
	img = cv2.bitwise_not(img)  # Black & White reversed 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img, gray 

def road_transform(gray): 
	dist = cv2.distanceTransform(gray, cv2.DIST_L2, 3)  # get distanceTransform map 
	dist_gauss = ndimage.gaussian_laplace(dist, sigma=3)  # apply gaussian_laplace filter 
	return dist, dist_gauss	

def local_min(dist_gauss):
	xy = np.argwhere(dist_gauss!=0)   # only consider the roads arranges 
	xy_split = np.split(xy,(np.argwhere(abs(np.diff(xy[:,1].flatten()))>1).flatten()+1))  # grouping the gaussian distribution (nonzero)   >> y= xy[:,1].flatten()
	xy_min = []   # list for gaussian distribution local minimum 
	for i in range(len(xy_split)):
		xy_min.append(xy_split[i][np.argmin(dist_gauss[xy_split[i][:,0],xy_split[i][:,1]])])    # append local minimum to list 
	return xy_min

def save_width_img(dist, xy_min):
	img = Image.open('C:\\Users\kbd_win_server\Desktop\getsize\getsize\ws5.jpg')
	draw = ImageDraw.Draw(img)
	font = ImageFont.truetype("arial.ttf", 8)
	for i in range(round(len(xy_min))):
		draw.text((xy_min[i][1],xy_min[i][0]), str(2*dist[xy_min[i][0]][xy_min[i][1]]), fill='red' ,font=font,anchor=None)    # draw text to image // width : 2*dist[xy_min[i][0]][xy_min[i][1]]
	img.save('width_img_output_fin.jpg')

if __name__ == '__main__':
	img, gray = initialize()
	dist, dist_gauss = road_transform(gray)
	xy_min = local_min(dist_gauss)
	save_width_img(dist, xy_min)





