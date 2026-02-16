import os, sys, argparse, math
import datetime

from ultralytics import YOLO
import os
import cv2
import numpy as np
from osgeo import gdal, osr, ogr
import pickle

def read_img(filename):
	dataset=gdal.Open(filename)

	im_width = dataset.RasterXSize
	im_height = dataset.RasterYSize

	im_geotrans = dataset.GetGeoTransform()
	im_proj = dataset.GetProjection()
	im_data = dataset.ReadAsArray(0,0,im_width,im_height)

	# del dataset 
	return im_width, im_height, im_proj, im_geotrans, im_data, dataset

def write_img(filename,im_proj,im_geotrans,im_data):
	if 'int8' in im_data.dtype.name:
		datatype = gdal.GDT_Byte
	elif 'int16' in im_data.dtype.name:
		datatype = gdal.GDT_UInt16
	else:
		datatype = gdal.GDT_Float32

	if len(im_data.shape) == 3:
		im_bands, im_height, im_width = im_data.shape
	else:
		im_bands, (im_height, im_width) = 1,im_data.shape 

	driver = gdal.GetDriverByName("GTiff")
	dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

	dataset.SetGeoTransform(im_geotrans)
	dataset.SetProjection(im_proj)

	if im_bands == 1:
		dataset.GetRasterBand(1).WriteArray(im_data)
	else:
		for i in range(im_bands):
			dataset.GetRasterBand(i+1).WriteArray(im_data[i])


def clip_tiff(pred_imgfile, small_tiff_savefolder, imgsz, stride):
	new_width = imgsz
	im_width, im_height, im_proj, geo_transform, im_data, dataset = read_img(pred_imgfile)
	new_w = im_width
	new_h = im_height
	extent_data = im_data

	count = 0
	i = 0
	num_ = 0
	name = os.path.split(pred_imgfile)[1]
	filename, ext = os.path.splitext(name)

	while i in range(new_h):
		j=0
		if (new_h-i) >=new_width:
			while j in range(new_w):          
				if (new_w-j) >=new_width:
					num_=num_+1
					im_data_m=extent_data[:,i:i+new_width,j:j+new_width]
					patch_path = os.path.join(small_tiff_savefolder, filename + '_' + str(num_) + '.tif')

					xmin = j
					ymin = i
					xmax = j+new_width
					ymax = i+new_width

					x1 = geo_transform[0] + xmin * geo_transform[1] + ymin * geo_transform[2]
					y1 = geo_transform[3] + xmin * geo_transform[4] + ymin * geo_transform[5]

					x2 = geo_transform[0] + xmax * geo_transform[1] + ymax * geo_transform[2]
					y2 = geo_transform[3] + xmax * geo_transform[4] + ymax * geo_transform[5]
					new_geo_transform = (x1, geo_transform[1], geo_transform[2], y1, geo_transform[4], geo_transform[5])
					write_img(patch_path, im_proj, new_geo_transform, im_data_m)
					j=j+stride
					
				if (new_w-j) <new_width:
					num_=num_+1
					im_data_m=extent_data[:,i:i+new_width,new_w-new_width:new_w]
					patch_path = os.path.join(small_tiff_savefolder, filename + '_' + str(num_) + '.tif')

					xmin = new_w-new_width
					ymin = i
					xmax = new_w
					ymax = i+new_width

					x1 = geo_transform[0] + xmin * geo_transform[1] + ymin * geo_transform[2]
					y1 = geo_transform[3] + xmin * geo_transform[4] + ymin * geo_transform[5]

					x2 = geo_transform[0] + xmax * geo_transform[1] + ymax * geo_transform[2]
					y2 = geo_transform[3] + xmax * geo_transform[4] + ymax * geo_transform[5]
					new_geo_transform = (x1, geo_transform[1], geo_transform[2], y1, geo_transform[4], geo_transform[5])

					write_img(patch_path, im_proj, new_geo_transform, im_data_m)
					j=new_w+1
					
			i=i+stride

		else :
			while j in range(new_w):
				if  (new_w-j) >=new_width:
					num_=num_+1
					im_data_m=extent_data[:,new_h-new_width:new_h,j:j+new_width]
					patch_path = os.path.join(small_tiff_savefolder, filename + '_' + str(num_) + '.tif')

					xmin = j
					ymin = new_h-new_width
					xmax = j+new_width
					ymax = new_h

					x1 = geo_transform[0] + xmin * geo_transform[1] + ymin * geo_transform[2]
					y1 = geo_transform[3] + xmin * geo_transform[4] + ymin * geo_transform[5]

					x2 = geo_transform[0] + xmax * geo_transform[1] + ymax * geo_transform[2]
					y2 = geo_transform[3] + xmax * geo_transform[4] + ymax * geo_transform[5]
					new_geo_transform = (x1, geo_transform[1], geo_transform[2], y1, geo_transform[4], geo_transform[5])

					write_img(patch_path, im_proj, new_geo_transform, im_data_m)
					j=j+stride
					
				if (new_w-j) <new_width:
					num_=num_+1
					im_data_m=extent_data[:,new_h-new_width:new_h,new_w-new_width:new_w]
					patch_path = os.path.join(small_tiff_savefolder, filename + '_' + str(num_) + '.tif')

					xmin = new_w-new_width
					ymin = new_h-new_width
					xmax = new_w
					ymax = new_h

					x1 = geo_transform[0] + xmin * geo_transform[1] + ymin * geo_transform[2]
					y1 = geo_transform[3] + xmin * geo_transform[4] + ymin * geo_transform[5]

					x2 = geo_transform[0] + xmax * geo_transform[1] + ymax * geo_transform[2]
					y2 = geo_transform[3] + xmax * geo_transform[4] + ymax * geo_transform[5]
					new_geo_transform = (x1, geo_transform[1], geo_transform[2], y1, geo_transform[4], geo_transform[5])
					write_img(patch_path, im_proj, new_geo_transform, im_data_m)
					j=new_w+1               
			i=new_h+1

def from_img2_geo(point, geotransform):

	point_x = point[0]
	point_y = point[1]  

	# 计算地理坐标
	point_geo_x = geotransform[0] + point_x * geotransform[1] + point_y * geotransform[2]
	point_geo_y = geotransform[3] + point_x * geotransform[4] + point_y * geotransform[5]

	return point_geo_x, point_geo_y

def from_geo2_img(point_geo_x, point_geo_y, geotransform):

	pixel_img_x = float((point_geo_x - geotransform[0]) / geotransform[1]) #float
	pixel_img_y = float((point_geo_y - geotransform[3]) / geotransform[5])

	return pixel_img_x, pixel_img_y

def convert2shp_small(output_folder_shp, image_path, boxes, seg_point_xy):  
	
	if not os.path.exists(output_folder_shp):
		os.mkdir(output_folder_shp)
	
	filepath, filename = os.path.split(image_path)
	shp_file_name = filename.replace('.tif', '.shp')
	shp_file_path = os.path.join(output_folder_shp, shp_file_name) #path to save shp

	dataset = gdal.Open(image_path)
	geotransform = dataset.GetGeoTransform()
	projection = dataset.GetProjection()

	srs = osr.SpatialReference()
	srs.ImportFromWkt(projection)
	ct = osr.CoordinateTransformation(srs, srs.CloneGeogCS())


	driver = ogr.GetDriverByName('ESRI Shapefile')
	data_source = driver.CreateDataSource(shp_file_path)


	layer = data_source.CreateLayer('polygon_layer', srs, geom_type=ogr.wkbPolygon)

	field_defn_id = ogr.FieldDefn('id', ogr.OFTInteger)
	layer.CreateField(field_defn_id)


	field_defn_clsid = ogr.FieldDefn('class', ogr.OFTInteger)
	layer.CreateField(field_defn_clsid)

	field_defn_name = ogr.FieldDefn('scores', ogr.OFTReal)
	layer.CreateField(field_defn_name)

	field_defn_name = ogr.FieldDefn('species', ogr.OFTString)
	field_defn_name.SetWidth(50) 
	layer.CreateField(field_defn_name)

	tree_classes = ['pine', 'spruce', 'birch', 'aspen']

	# read the predicted results
	boxes_conf = boxes.conf.tolist()
	box_cls = boxes.cls.tolist() # Class labels for each box.
	
	for j in range(len(boxes)):   
		obj_conf = boxes_conf[j]
		obj_cls_id = str(int(box_cls[j]))
		obj_cls_name = tree_classes[int(box_cls[j])]
		points = seg_point_xy[j]

		if len(points) > 0:
			
			start_point = points[0]
			start_point_x, start_point_y = from_img2_geo(start_point, geotransform) 

			feature = ogr.Feature(layer.GetLayerDefn())
			polygon = ogr.Geometry(ogr.wkbPolygon)     
			ring = ogr.Geometry(ogr.wkbLinearRing)

			for point in points:
				point_geo_x, point_geo_y = from_img2_geo(point, geotransform) 
				ring.AddPoint(point_geo_x, point_geo_y)
			
			ring.AddPoint(start_point_x, start_point_y) # add the start point
			polygon.AddGeometry(ring)
			
			#write attribute table
			feature.SetGeometry(polygon)
			feature.SetField('id', j+1)
			feature.SetField('class', obj_cls_id)
			feature.SetField('scores', obj_conf)
			feature.SetField('species', obj_cls_name)
			layer.CreateFeature(feature)

	data_source = None

def convert2shp_large(output_folder_lshp, l_tiff_imgfile, pred_results):
	
	if not os.path.exists(output_folder_lshp):
		os.mkdir(output_folder_lshp)
	
	filepath, filename = os.path.split(l_tiff_imgfile)
	shp_file_name = filename.replace('.tif', '.shp')
	shp_file_path = os.path.join(output_folder_lshp, shp_file_name) #path to save shp

	dataset_l = gdal.Open(l_tiff_imgfile)

	geotransform_l = dataset_l.GetGeoTransform()
	projection_l = dataset_l.GetProjection()


	srs = osr.SpatialReference()
	srs.ImportFromWkt(projection_l)
	ct = osr.CoordinateTransformation(srs, srs.CloneGeogCS())

	driver = ogr.GetDriverByName('ESRI Shapefile')
	data_source = driver.CreateDataSource(shp_file_path)


	layer = data_source.CreateLayer('polygon_layer', srs, geom_type=ogr.wkbPolygon)

	field_defn_id = ogr.FieldDefn('id', ogr.OFTInteger)
	layer.CreateField(field_defn_id)

	field_defn_clsid = ogr.FieldDefn('class', ogr.OFTInteger)
	layer.CreateField(field_defn_clsid)

	field_defn_name = ogr.FieldDefn('scores', ogr.OFTReal)
	layer.CreateField(field_defn_name)

	field_defn_name = ogr.FieldDefn('species', ogr.OFTString)
	field_defn_name.SetWidth(50) 
	layer.CreateField(field_defn_name)

	tree_classes = ['pine', 'spruce', 'birch', 'aspen']	


	for patch_path in pred_results.keys():

		dataset_s = gdal.Open(patch_path) #key: small tiff path
		geotransform_s = dataset_s.GetGeoTransform()
		projection_s = dataset_s.GetProjection()

		pred_patch = pred_results[patch_path]

		boxes = pred_patch[0].boxes
		masks = pred_patch[0].masks    # Masks object for segmentation masks outputs
		seg_point_xy = masks.xy        # A list of segments in pixel coordinates

		boxes_conf = boxes.conf.tolist()
		box_cls = boxes.cls.tolist() # Class labels for each box.

		for j in range(len(boxes)):   
			obj_conf = boxes_conf[j]
			obj_cls_id = str(int(box_cls[j]))
			obj_cls_name = tree_classes[int(box_cls[j])]
			points = seg_point_xy[j]

			if len(points) > 0:			
				start_point = points[0]
				start_point_x, start_point_y = from_img2_geo(start_point, geotransform_s) 

				feature = ogr.Feature(layer.GetLayerDefn())
				polygon = ogr.Geometry(ogr.wkbPolygon)     
				ring = ogr.Geometry(ogr.wkbLinearRing)

				for point in points:
					point_geo_x, point_geo_y = from_img2_geo(point, geotransform_s) 
					ring.AddPoint(point_geo_x, point_geo_y)
				
				ring.AddPoint(start_point_x, start_point_y) # add the start point
				polygon.AddGeometry(ring)
				
				#write attribute table
				feature.SetGeometry(polygon)
				feature.SetField('id', j+1)
				feature.SetField('class', obj_cls_id)
				feature.SetField('scores', obj_conf)
				feature.SetField('species', obj_cls_name)
				layer.CreateFeature(feature)

			dataset_s = None

	data_source = None
	dataset_l = None

def start_nms(pred_pine, pred_spruce, pred_birch, pred_aspen):
	all_index = []
	iou_thre = 0.3
	#with open('./PRED_ALL/pred_pine2.pkl', "rb") as pickle_file1:
		#pred_pine = pickle.load(pickle_file1)
	print('before nms len(pred_pine):', len(pred_pine))
	pred_pine_nms = non_max_suppress(pred_pine, iou_thre)
	print('after nms len(pred_pine):', len(pred_pine_nms))
	if len(pred_pine_nms)>0:
		pine_index = pred_pine_nms[:, 5] #float
		pine_index_int = [int(x) for x in pine_index]
		all_index.extend(pine_index_int)

	#with open('./PRED_ALL/pred_spruce2.pkl', "rb") as pickle_file2:
		#pred_spruce = pickle.load(pickle_file2)
	print('before nms len(pred_spruce):', len(pred_spruce))
	pred_spruce_nms = non_max_suppress(pred_spruce, iou_thre)
	if len(pred_spruce_nms)>0:
		spruce_index = pred_spruce_nms[:, 5]
		spruce_index_int = [int(x) for x in spruce_index]
		all_index.extend(spruce_index_int)

	print('after nms len(pred_spruce):', len(pred_spruce_nms))

	pred_birch_nms = non_max_suppress(pred_birch, iou_thre)
	if len(pred_birch_nms)>0:
		birch_index = pred_birch_nms[:, 5]
		birch_index_int = [int(x) for x in birch_index]
		all_index.extend(birch_index_int)
	print('before nms len(pred_birch):', len(pred_birch))
	print('after nms len(pred_birch):', len(pred_birch_nms))

	#with open('./PRED_ALL/pred_aspen2.pkl', "rb") as pickle_file4:
		#pred_aspen = pickle.load(pickle_file4)
	pred_aspen_nms = non_max_suppress(pred_aspen, iou_thre)
	
	if len(pred_aspen_nms)>0:
		aspen_index = pred_aspen_nms[:, 5]
		pred_aspen_nms = pred_aspen_nms.tolist()
		aspen_index_int = [int(x) for x in aspen_index]
		all_index.extend(aspen_index_int)

	print('before nms len(pred_aspen):', len(pred_aspen))
	print('after nms len(pred_aspen):', len(pred_aspen_nms))

	#with open('./PRED_ALL/all_index_nms2.pkl', "wb") as pickle_file6:
		#pickle.dump(all_index, pickle_file6)

	return all_index

def non_max_suppress(predicts_dict, threshold):
	
	bbox_array = np.array(predicts_dict, dtype=np.float64)

	if len(bbox_array) > 0:

		x1 = bbox_array[:, 0]
		y1 = bbox_array[:, 1]
		x2 = bbox_array[:, 2]
		y2 = bbox_array[:, 3]
		scores = bbox_array[:, 4]
		order = scores.argsort()[::-1]
		areas = (x2 - x1 + 1) * (y2 - y1 + 1)
		keep = []
		
		while order.size > 0:
			i = order[0]
			keep.append(i)
			xx1 = np.maximum(x1[i], x1[order[1:]])
			yy1 = np.maximum(y1[i], y1[order[1:]])
			xx2 = np.minimum(x2[i], x2[order[1:]])
			yy2 = np.minimum(y2[i], y2[order[1:]])
			inter = np.maximum(0.0, xx2-xx1+1) * np.maximum(0.0, yy2-yy1+1)
			iou = inter / (areas[i] + areas[order[1:]] - inter)
			inds = np.where(iou <= threshold)[0] 
			order = order[inds + 1]

		bbox = bbox_array[keep]
		
		#predicts_dict[object_name] = bbox.tolist()
		#predicts_dict = predicts_dict
	else:
		bbox = bbox_array

	return bbox

def convert2shp_large_nms(new_pred_results, all_index_nms, l_tiff_imgfile, output_folder_lshp):

	if not os.path.exists(output_folder_lshp):
		os.mkdir(output_folder_lshp)

	#with open(re_s_pkl, "rb") as pred_results_file: 
		#new_pred_results = pickle.load(pred_results_file)

	#with open('./PRED_ALL/all_index_nms2.pkl', "rb") as pickle_file5:
		#all_index_nms = pickle.load(pickle_file5)
	
	filepath, filename = os.path.split(l_tiff_imgfile)
	shp_file_name = filename.replace('.tif', '.shp')
	shp_file_path = os.path.join(output_folder_lshp, shp_file_name) #path to save shp

	dataset_l = gdal.Open(l_tiff_imgfile)

	geotransform_l = dataset_l.GetGeoTransform()
	projection_l = dataset_l.GetProjection()

	srs = osr.SpatialReference()
	srs.ImportFromWkt(projection_l)
	ct = osr.CoordinateTransformation(srs, srs.CloneGeogCS())

	driver = ogr.GetDriverByName('ESRI Shapefile')
	data_source = driver.CreateDataSource(shp_file_path)

	layer = data_source.CreateLayer('polygon_layer', srs, geom_type=ogr.wkbPolygon)

	field_defn_id = ogr.FieldDefn('id', ogr.OFTInteger)
	layer.CreateField(field_defn_id)

	field_defn_clsid = ogr.FieldDefn('class', ogr.OFTInteger)
	layer.CreateField(field_defn_clsid)

	field_defn_name = ogr.FieldDefn('scores', ogr.OFTReal)
	layer.CreateField(field_defn_name)

	field_defn_name = ogr.FieldDefn('species', ogr.OFTString)
	field_defn_name.SetWidth(50)  # 设置字段宽度
	layer.CreateField(field_defn_name)

	tree_classes = ['pine', 'spruce', 'birch', 'aspen']	

	obj_id = 0
	for common_index in all_index_nms:
		pred_feature = new_pred_results[common_index] #[patch_path, box_points, points, obj_cls_id, obj_conf] #save for index

		patch_path = pred_feature[0]
		points = pred_feature[2]
		obj_cls_id = pred_feature[3]
		obj_conf = pred_feature[4]
		obj_cls_name = tree_classes[int(obj_cls_id)]

		dataset_s = gdal.Open(patch_path) #key: small tiff path
		geotransform_s = dataset_s.GetGeoTransform()
		projection_s = dataset_s.GetProjection()
		
		if len(points) >0:
			start_point = points[0]
			start_point_x, start_point_y = from_img2_geo(start_point, geotransform_s) 

			feature = ogr.Feature(layer.GetLayerDefn())
			polygon = ogr.Geometry(ogr.wkbPolygon)     
			ring = ogr.Geometry(ogr.wkbLinearRing)

			for point in points:
				point_geo_x, point_geo_y = from_img2_geo(point, geotransform_s) 
				ring.AddPoint(point_geo_x, point_geo_y)
			
			ring.AddPoint(start_point_x, start_point_y) # add the start point
			polygon.AddGeometry(ring)
			obj_id += 1			
			#write attribute table
			feature.SetGeometry(polygon)
			feature.SetField('id', obj_id)
			feature.SetField('class', obj_cls_id)
			feature.SetField('scores', obj_conf)
			feature.SetField('species', obj_cls_name)
			layer.CreateFeature(feature)

		dataset_s = None

	
	data_source = None
	dataset_l = None

def pred_tiff_s(small_tiff_savefolder, output_folder_sshp, re_s_pkl):

	if not os.path.exists(output_folder_sshp):
		os.mkdir(output_folder_sshp)
	
	pred_results = {}
	for s_tiff in os.listdir(small_tiff_savefolder):
		s_tiff_imgfile = os.path.join(small_tiff_savefolder, s_tiff)

		#predict
		# Run batched inference on a list of images
		results = model(s_tiff_imgfile, save_txt=False, imgsz=128, save_conf=False, conf=0.5)  
		# return a list of Results objects
		#print(results)
		boxes = results[0].boxes
		masks = results[0].masks    # Masks object for segmentation masks outputs

		if len(boxes) >0 and len(masks)>0:
			seg_point_xy = masks.xy # A list of segments in pixel coordinates 

			# save result of small tiff in shpfile format
			convert2shp_small(output_folder_sshp, s_tiff_imgfile, boxes, seg_point_xy)
			#pred_results.append(results)
			pred_results[s_tiff_imgfile] = results

	with open(re_s_pkl, "wb") as pickle_file:
		pickle.dump(pred_results, pickle_file)

	#reture presicted results of small patches
	#return pred_results

def convert_tiffs2l(re_s_pkl, l_tiff_imgfile):
	# 从 pickle 文件中加载数据 small tiff results
	with open(re_s_pkl, "rb") as pickle_file:
		pred_results = pickle.load(pickle_file)

	pred_results_pine = []  #[ [x1, y1, x2, y2, score, clsid, [mask]], ..., [] ]
	pred_results_spruce = []  
	pred_results_birch = []  
	pred_results_aspen = [] 

	dataset_l = gdal.Open(l_tiff_imgfile) #key: small tiff path
	geotransform_l = dataset_l.GetGeoTransform()
	projection_l = dataset_l.GetProjection()

	new_pred_results = {}
	box_index = 0

	for patch_path in pred_results.keys():
		
		dataset_s = gdal.Open(patch_path) #key: small tiff path
		geotransform_s = dataset_s.GetGeoTransform()
		projection_s = dataset_s.GetProjection()

		pred_patch = pred_results[patch_path]
		boxes = pred_patch[0].boxes
		box_point_xy = boxes.xyxy.tolist() # Boxes in [x1, y1, x2, y2] format.

		masks = pred_patch[0].masks    # Masks object for segmentation masks outputs
		seg_point_xy = masks.xy        # A list of segments in pixel coordinates

		boxes_conf = boxes.conf.tolist()
		box_cls = boxes.cls.tolist() # Class labels for each box.

		for j in range(len(boxes)): 
			pred_feature = [] # save boxes for nms
			pred_results_index = [] # save boxes for index
			
			obj_conf = boxes_conf[j]
			obj_cls_id = str(int(box_cls[j])) #int 
			#obj_cls_name = tree_classes[int(box_cls[j])]
			points = seg_point_xy[j]  #mask points
			box_points = box_point_xy[j]
			pred_results_index.extend([patch_path, box_points, points, obj_cls_id, obj_conf]) #save for index
		
			if len(box_points) > 0:
				new_pred_results[box_index] = pred_results_index
				
				point_geo_x1, point_geo_y1 = from_img2_geo([box_points[0], box_points[1]], geotransform_s) # on small tiff, geo coor
				point_geo_x2, point_geo_y2 = from_img2_geo([box_points[2], box_points[3]], geotransform_s) # on small tiff, geo coor

				point_img_x1, point_img_y1 = from_geo2_img(point_geo_x1, point_geo_y1, geotransform_l) # on large tiff, image coor
				point_img_x2, point_img_y2 = from_geo2_img(point_geo_x2, point_geo_y2, geotransform_l) 

				pred_feature.extend([point_img_x1, point_img_y1, point_img_x2, point_img_y2])
				pred_feature.append(obj_conf)
				pred_feature.append(box_index)

				box_index += 1

				if obj_cls_id == '0':
					pred_results_pine.append(pred_feature)
					#cv2.polylines(image, [point_list], isClosed=True, color=(102, 102, 255), thickness=1) #BGR red
				
				elif obj_cls_id == '1':
					pred_results_spruce.append(pred_feature)
					#cv2.polylines(image, [point_list], isClosed=True, color=(255, 102, 255), thickness=1) #pink
				
				elif obj_cls_id == '2':
					pred_results_birch.append(pred_feature)
					#cv2.polylines(image, [point_list], isClosed=True, color=(0, 128, 255), thickness=1) #orange
				
				elif obj_cls_id == '3':
					pred_results_aspen.append(pred_feature)
					#cv2.polylines(image, [point_list], isClosed=True, color=(255, 255, 102), thickness=1) #blue

		dataset_s = None


	
	return new_pred_results, pred_results_pine, pred_results_spruce, pred_results_birch, pred_results_aspen
	


if __name__ == '__main__':


	PRED_rootfolder =  './site1/'
	pred_image_folder = './site1/input_l_5ch_tiff' 
	output_folder_sshp = './site1/output_s_shp_cu128' # predict results of small tiff
	if not os.path.exists(output_folder_sshp):
			os.mkdir(output_folder_sshp)

	output_folder_lshp = './site1/output_nms_l_shp_cu128' # predict results of small tiff after applying NMS
	if not os.path.exists(output_folder_lshp):
			os.mkdir(output_folder_lshp)


	# clip tiff with overlapping
	imgsz=128
		
	for l_tiff in os.listdir(pred_image_folder):
	
		small_tiff_savefolder = os.path.join(PRED_rootfolder, 'clip_s_tiff') # 32 pixels

		if not os.path.exists(small_tiff_savefolder):
			os.mkdir(small_tiff_savefolder)

		# 1. Clip large tiff to small size 
		l_tiff_imgfile = os.path.join(pred_image_folder, l_tiff)
		clip_tiff(l_tiff_imgfile, small_tiff_savefolder, imgsz=128, stride=64)

		# Supre-resolution
		# output_SR_tiff_savefolder = os.path.join(PRED_rootfolder, 'CU_s_tiff_32_128_5ch')
		
		#if not os.path.exists(output_SR_tiff_savefolder):
		#	os.mkdir(output_SR_tiff_savefolder)
		#scale_factor = 4
		#SR_tiff_s(small_tiff_savefolder, output_SR_tiff_savefolder, scale_factor)

		# predict small tiff and output results
		# Load a model
		model = YOLO('./word_dir/train/weights/best.pt')
		# pklfile to save all result of small tiffs 
		pkl_folder = './site1/pklfile_X4_128'
		
		if not os.path.exists(pkl_folder):
			os.mkdir(pkl_folder)
		re_s_pkl = os.path.join(pkl_folder, l_tiff.replace('.tif', '.pkl')) 
		
		pred_tiff_s(small_tiff_savefolder, output_folder_sshp, re_s_pkl)

		#convert predicted results to shp without nms
		#convert2shp_large_nms(l_tiff_imgfile)

		# convert small img coordinate of predicted boxes to large geo, to large img for nms
		new_pred_results, pred_pine, pred_spruce, pred_birch, pred_aspen = convert_tiffs2l(re_s_pkl, l_tiff_imgfile)

		# optional: start nms
		all_index_nms = start_nms(pred_pine, pred_spruce, pred_birch, pred_aspen)

		# convert nms results to shp, large 
		convert2shp_large_nms(new_pred_results, all_index_nms, l_tiff_imgfile, output_folder_lshp)

		# postprocess lagre shp and visualize them in circl





	
