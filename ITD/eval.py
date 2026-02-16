from ultralytics import YOLO

# Load a model
model = YOLO('./word_dir/train/weights/best.pt')  # load a custom model


# Customize validation settings
#metrics = model.val(data='./Datasets/tree_seg.yaml ')
metrics = model.val()

# Validate the model
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

metrics.seg.map    # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps   # a list contains map52	0-95(M) of each category
