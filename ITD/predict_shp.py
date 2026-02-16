
from ultralytics import YOLO
import cv2
from osgeo import gdal, osr

import geopandas as gpd
from shapely.geometry import Polygon
import os
import cv2
import numpy as np


def get_crs_from_tiff(tiff_path):
    dataset = gdal.Open(tiff_path, gdal.GA_ReadOnly)
    proj = dataset.GetProjection()
    srs = osr.SpatialReference(wkt=proj)
    return srs.ExportToWkt() if proj else None

def pixel_to_geo(transform, x_pixel, y_pixel):
    """Convert pixel (col, row) to geo coordinates using affine transform"""
    x_geo = transform[0] + x_pixel * transform[1] + y_pixel * transform[2]
    y_geo = transform[3] + x_pixel * transform[4] + y_pixel * transform[5]
    return (x_geo, y_geo)

def cv2_draw(boxes, masks, imgpath, img_file, img_ch):
    boxes_conf = boxes.conf.tolist()
    box_cls = boxes.cls.tolist()
    seg_point_xy = masks.xy

    # Load image for visualization
    if img_ch == 3:
        image = cv2.imread(imgpath)
    else:
        dataset = gdal.Open(imgpath, gdal.GA_ReadOnly)
        band_red = dataset.GetRasterBand(1).ReadAsArray()
        band_green = dataset.GetRasterBand(2).ReadAsArray()
        band_blue = dataset.GetRasterBand(3).ReadAsArray()
        bgr_img = np.dstack((band_blue, band_green, band_red))
        image = bgr_img

    # Prepare save paths
    shp_save_dir = './shp_5ch/'
    os.makedirs(shp_save_dir, exist_ok=True)
    shp_save_path = os.path.join(shp_save_dir, img_file.replace('.tif', '.shp'))

    img_save_dir = './image_5ch/'
    os.makedirs(img_save_dir, exist_ok=True)
    img_save_path = os.path.join(img_save_dir, img_file)

    # Get transform and CRS from GDAL
    dataset = gdal.Open(imgpath, gdal.GA_ReadOnly)
    transform = dataset.GetGeoTransform()
    crs_wkt = get_crs_from_tiff(imgpath)

    polygons = []
    labels = []
    confs = []

    for j in range(len(boxes)):
        obj_conf = boxes_conf[j]
        obj_cls_id = int(box_cls[j])
        points = seg_point_xy[j]

        # Convert all points to geo coordinates
        geo_point_list = [pixel_to_geo(transform, p[0], p[1]) for p in points]

        if len(geo_point_list) < 3:
            continue
        polygons.append(Polygon(geo_point_list))
        labels.append(obj_cls_id)
        confs.append(obj_conf)

        # Also draw polygon in pixel space
        point_np = np.array([(int(p[0]), int(p[1])) for p in points], dtype=np.int32)
        color = {
            0: (0, 255, 128),
            1: (255, 120, 178),
            2: (0, 128, 255),
            3: (0, 255, 255),
        }.get(obj_cls_id, (255, 255, 255))
        cv2.polylines(image, [point_np], isClosed=True, color=color, thickness=1)

    # Save image
    #cv2.imwrite(img_save_path, image)
    #print(f'[✓] Saved prediction image: {img_save_path}')

    # Save shapefile with geo coordinates
    if polygons:
        gdf = gpd.GeoDataFrame({
            'class': labels,
            'conf': confs,
            'geometry': polygons
        })
        if crs_wkt:
            gdf.set_crs(crs_wkt, inplace=True)
        else:
            print("[!] CRS not found, shapefile saved without projection.")

        gdf.to_file(shp_save_path)
        print(f'[✓] Saved georeferenced shapefile: {shp_save_path}')



def Predict_img(model, imgpath, img_file, img_ch):

	# predict image
	results = model(imgpath, save_txt=False, imgsz=128, save_conf=False, conf=0.5)  # return a list of Results objects , channel=3

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