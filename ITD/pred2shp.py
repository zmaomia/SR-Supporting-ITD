#import geopandas as gpd
#import rasterio
#from rasterio.transform import from_origin
#from shapely.geometry import box
#import pyproj
import json
from osgeo import osr, gdal, ogr

# 读取遥感影像文件以获取地理参考信息
image_file = "./test/jokisalo_region1_deno.tif"

dataset = gdal.Open(image_file)

# 获取输入文件的地理转换信息和投影信息
geotransform = dataset.GetGeoTransform()
projection = dataset.GetProjection()

# 创建地理坐标转换器
srs = osr.SpatialReference()
srs.ImportFromWkt(projection)
ct = osr.CoordinateTransformation(srs, srs.CloneGeogCS())


# 创建一个新的shapefile
driver = ogr.GetDriverByName('ESRI Shapefile')
data_source = driver.CreateDataSource('./test_shp/test.shp')

# 创建一个新的图层
layer = data_source.CreateLayer('polygon_layer', srs, geom_type=ogr.wkbPolygon)


# 创建字段
field_defn_id = ogr.FieldDefn('id', ogr.OFTInteger)
layer.CreateField(field_defn_id)


field_defn_clsid = ogr.FieldDefn('class', ogr.OFTInteger)
layer.CreateField(field_defn_clsid)

field_defn_name = ogr.FieldDefn('scores', ogr.OFTReal)
layer.CreateField(field_defn_name)

field_defn_name = ogr.FieldDefn('species', ogr.OFTString)
field_defn_name.SetWidth(50)  # 设置字段宽度
layer.CreateField(field_defn_name)

tree_classes = ['pine', 'spruce', 'birch', 'populus']
pred_jsonfile = './test/jokisalo_region1_deno_1.json'

with open(pred_jsonfile) as pred_json:
    pred_results = json.load(pred_json)

pred_labels =  pred_results['labels']
pred_confidence = pred_results['scores']
pred_bboxes = pred_results['bboxes']


for tree_i, pre_tree in enumerate(pred_labels):
    tree_class_id = pre_tree
    tree_class = tree_classes[tree_class_id]

    tree_score = pred_confidence[tree_i]
    pred_bbox = pred_bboxes[tree_i]

    if tree_score > 0.3:

        xmin, ymin, xmax, ymax = pred_bbox # coordinates in image pixels.

        # 计算地理坐标
        minx = geotransform[0] + xmin * geotransform[1] + ymin * geotransform[2]
        miny = geotransform[3] + xmin * geotransform[4] + ymin * geotransform[5]
        #minlon, minlat, _ = ct.TransformPoint(minx, miny)


        maxx = geotransform[0] + xmax * geotransform[1] + ymax * geotransform[2]
        maxy = geotransform[3] + xmax * geotransform[4] + ymax * geotransform[5]
        #maxlon, maxlat, _ = ct.TransformPoint(maxx, maxy)

        feature = ogr.Feature(layer.GetLayerDefn())
        polygon = ogr.Geometry(ogr.wkbPolygon)
        
        ring = ogr.Geometry(ogr.wkbLinearRing)
        #ring.AddPoint(minlon, minlat)
        #ring.AddPoint(maxlon, maxlat)
        #ring.AddPoint(minlon, minlat)

        ring.AddPoint(minx, miny)
        ring.AddPoint(maxx, miny)
        ring.AddPoint(maxx, maxy)
        ring.AddPoint(minx, maxy)
        ring.AddPoint(minx, miny)


        polygon.AddGeometry(ring)
        feature.SetGeometry(polygon)
        feature.SetField('id', tree_i+1)
        feature.SetField('class', tree_class_id)
        feature.SetField('scores', tree_score)
        feature.SetField('species', tree_class)

        layer.CreateFeature(feature)

# 释放资源
data_source = None


    # 创建地理坐标的目标框
    #bbox = box(minlon, minlat, maxlon, maxlat)
    #gdf = gdf.concat({'class': tree_class_num, 'species':tree_class, 'score': tree_score, 'geometry': bbox}, ignore_index=True)

# 将 GeoDataFrame保存为 shapefile
#gdf.to_file('./test.shp')
    #convert coordinates from image to real world
    #minx, miny = transform * (xmin, ymin)
    #maxx, maxy = transform * (xmax, ymax)
    # 投影转换
    #minlon, minlat = project.transform(minx, miny)
    #maxlon, maxlat = project.transform(maxx, maxy)
    # 创建地理坐标的目标框
    #bbox = box(minlon, minlat, maxlon, maxlat)
    #gdf = gdf.append({'class': tree_class_num, 'species':tree_class, 'score': score, 'geometry': bbox}, ignore_index=True)

