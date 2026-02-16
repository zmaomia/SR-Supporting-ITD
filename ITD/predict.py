from ultralytics import YOLO
import cv2
import numpy as np
import os
from osgeo import gdal

def cv2_draw(boxes, masks, imgpath, img_file, img_ch):
	boxes_conf = boxes.conf.tolist() # prediected confidence for each box
	box_cls = boxes.cls.tolist() # Class labels for each box.
	seg_point_xy = masks.xy # A list of segments in pixel coordinates 

	if img_ch == 3:
		image = cv2.imread(imgpath)

	else:
		multiband_dataset = gdal.Open(imgpath, gdal.GA_ReadOnly)
		rgb_image = np.zeros((multiband_dataset.RasterYSize, multiband_dataset.RasterXSize, 3), dtype=np.uint8)
		band_red = multiband_dataset.GetRasterBand(1).ReadAsArray() #r
		band_green = multiband_dataset.GetRasterBand(2).ReadAsArray() # g
		band_blue = multiband_dataset.GetRasterBand(3).ReadAsArray() # b
		bgr_img = np.dstack((band_blue, band_green, band_red))
		image = bgr_img

	# save rgb image
	#cv2.imwrite(os.path.join('./PRED_ALL/image_rgb', img_file), image)

	for j in range(len(boxes)):	
		obj_conf = boxes_conf[j]
		obj_cls_id = str(int(box_cls[j])) #int 
		points = seg_point_xy[j]
		point_list = []
		
		for point in points:
			new_point = []
			new_point.append(int(point[0]))
			new_point.append(int(point[1]))
			point_list.append(new_point)
		point_list = np.array(point_list)

		if obj_cls_id == '0':
			cv2.polylines(image, [point_list], isClosed=True, color=(0, 255, 128), thickness=1) #BGR light green
		elif obj_cls_id == '1':
			cv2.polylines(image, [point_list], isClosed=True, color=(255, 120, 178), thickness=1) #purple
		elif obj_cls_id == '2':
			cv2.polylines(image, [point_list], isClosed=True, color=(0, 128, 255), thickness=1) #orange
		elif obj_cls_id == '3':
			cv2.polylines(image, [point_list], isClosed=True, color=(0, 255, 255), thickness=1) #yellow 
	
	print('saved results in images')
	cv2.imwrite(os.path.join('./image_5ch/', img_file), image)

def Predict_img(model, imgpath, img_file, img_ch):
	
	# predict image
	results = model(imgpath, save_txt=False, imgsz=512, save_conf=False, conf=0.5)  # return a list of Results objects , channel=3

	boxes = results[0].boxes
	print('len boxes:', len(boxes))
	masks = results[0].masks    # Masks object for segmentation masks outputs
	
	
	# visualize using cv2
	if len(boxes) > 0 and len(masks)>0:
		print('len masks:', len(masks))
		cv2_draw(boxes, masks, imgpath, img_file, img_ch)



if __name__ == '__main__':

	# Load a model
	#model = YOLO('yolov8x-seg.pt')  # load an official model
	model = YOLO('./word_dir/weights/best.pt')  # load a custom model
	img_ch = 5

	img_folder = './image_5ch/'

	for img_file in os.listdir(img_folder):
		imgpath = os.path.join(img_folder, img_file)
		Predict_img(model, imgpath, img_file, img_ch)

