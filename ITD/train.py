from ultralytics import YOLO

model = YOLO('yolov8x-seg.yaml').load('yolov8x-seg.pt')

# Train the model using the 'tree.yaml' dataset for 500 epochs
results = model.train(data='./Datasets/tree_seg.yaml ', 
	epochs=500,
	iou=0.5,
	imgsz=128, #32 #64 #128 #256
	batch=2,
	lr0=0.001, 
	lrf=0.001,
	hsv_h=0,  #.015  # (float) image HSV-Hue augmentation (fraction),
	hsv_s=0,  #.7  # (float) image HSV-Saturation augmentation (fraction),
	hsv_v=0, #.4  # (float) image HSV-Value augmentation (fraction)
	workers=1,
	project='./word_dir/',
	device=0)

# Evaluate the model's performance on the validation set
#results = model.val()

# Perform object detection on an image using the model
#results = model('https://ultralytics.com/images/bus.jpg')

